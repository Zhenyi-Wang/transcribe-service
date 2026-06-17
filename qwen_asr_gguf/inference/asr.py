# coding=utf-8
import os
import sys
import time
import re
import codecs
import dataclasses
import numpy as np
import multiprocessing as mp
from pathlib import Path
from collections import deque
from typing import Optional, List

from .schema import MsgType, StreamingMessage, DecodeResult, ASREngineConfig, TranscribeResult, ForcedAlignItem, ForcedAlignResult
from .utils import normalize_language_name, validate_language
from .encoder import QwenAudioEncoder
from . import llama
from . import logger

@dataclasses.dataclass
class ASRS_Segment:
    """管理分片记忆及其物理时间坐标"""
    idx: int
    audio_start: float
    audio_end: float
    text: str = ""
    items: List[ForcedAlignItem] = None   

class QwenASREngine:
    """Qwen3-ASR 流式转录引擎 (GGUF 后端) - 统一辅助进程架构"""
    def __init__(self, config: ASREngineConfig):
        self.config = config
        self.verbose = config.verbose
        if self.verbose: print(f"--- [QwenASR] 初始化引擎 (Provider: {config.onnx_provider}) ---")
        
        # 路径解析
        llm_gguf = os.path.join(config.model_dir, config.llm_fn)
        frontend_path = os.path.join(config.model_dir, config.encoder_frontend_fn)
        backend_path = os.path.join(config.model_dir, config.encoder_backend_fn)

        # 1. 初始化 Encoder
        self.encoder = QwenAudioEncoder(
            frontend_path=frontend_path,
            backend_path=backend_path,
            onnx_provider=config.onnx_provider,
            dml_pad_to=config.dml_pad_to,
            verbose=self.verbose
        )

        # 2. 初始化 Aligner (可选)
        self.aligner = None
        if config.enable_aligner and config.align_config:
            from .aligner import QwenForcedAligner
            self.aligner = QwenForcedAligner(config.align_config)
        
        # 3. 加载识别 LLM
        self.model = llama.LlamaModel(llm_gguf, use_gpu=config.llm_use_gpu)
        self.embedding_table = llama.get_token_embeddings_gguf(llm_gguf)

        # 动态计算 n_batch 和 n_ubatch：Qwen3 多平面位置编码需要 pos_arr = total_len × 4
        # 每秒音频约 13 embedding frames，需覆盖 (memory_num + 1) 个 chunk + prefix/suffix 余量
        frames_per_chunk = int(config.chunk_size * 13)
        total_max_frames = frames_per_chunk * (config.memory_num + 1) + 500  # 500 for prefix/suffix tokens
        # n_batch 需要覆盖 pos_arr 长度 (total_len × 4)
        n_batch = max(4096, ((total_max_frames * 4 + 4095) // 4096) * 4096)  # 向上对齐到 4096
        # n_ubatch 覆盖 total_len
        n_ubatch = max(512, ((total_max_frames + 511) // 512) * 512)  # 向上对齐到 512
        # causal attention (attention_type=0)：Qwen3-ASR 需要 causal attention（实测验证）
        # causal 模式下 llama.cpp 会将 n_batch 限制到 min(n_ctx, params.n_batch)
        # 因此 n_ctx 必须足够大以容纳 total_len × 4（mrope 四平面位置编码）
        required_ctx = n_batch + 4096  # n_batch 已经是 total_max_frames * 4 向上对齐
        if config.n_ctx < required_ctx:
            config.n_ctx = required_ctx
        self.ctx = llama.LlamaContext(self.model, n_ctx=config.n_ctx, n_batch=n_batch, n_ubatch=n_ubatch, embeddings=False, attention_type=0)
        # 诊断用：记录 context 容量，供 _decode 越界检查（GGML get_rows 索引越界定位）
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_ctx = config.n_ctx

        # 缓存 Token ID
        self.ID_IM_START = self.model.token_to_id("<|im_start|>")
        self.ID_IM_END = self.model.token_to_id("<|im_end|>")
        self.ID_AUDIO_START = self.model.token_to_id("<|audio_start|>")
        self.ID_AUDIO_END = self.model.token_to_id("<|audio_end|>")
        self.ID_ASR_TEXT = self.model.token_to_id("<asr_text>")
        self._cancel_event = None  # 可选外部取消信号（threading.Event），asr() 在 chunk 边界检查；None=不可取消（默认，向后兼容）

    def set_cancel_event(self, event):
        """设置/清除外部取消信号。asr() 在每个 chunk 边界检查 event.is_set()。
        event=None 清除（恢复默认不可取消）。transcribe-service 等不调此方法的调用方行为不变。"""
        self._cancel_event = event

    def shutdown(self):
        # 主动断引用，触发各 __del__ → llama_free / llama_model_free（确定性释放，不等 GC）。
        # llama.py 已有：LlamaModel.__del__→llama_model_free、LlamaContext.__del__→llama_free。
        # aligner/encoder 持有自己的 model/ctx/ONNX session，断引用后由其 __del__/GC 级联释放。
        self.ctx = None
        self.model = None
        self.aligner = None
        self.encoder = None
        self.embedding_table = None
        if self.verbose: print("--- [QwenASR] 引擎已关闭 ---")

    def _build_prompt_embd(self, audio_embd: np.ndarray, prefix_text: str, context: Optional[str], language: Optional[str]):
        """构造用于 LLM 输入的 Embedding 序列 (区块化打包模式)

        language 为 None 时进入自动检测模式：不添加 <asr_text> 到 prompt，
        模型会自行输出 "language X\\n<asr_text>text" 格式，由 _parse_asr_output 解析。
        language 有值时进入强制模式：prompt 中包含 "language X<asr_text>"，
        模型直接输出纯文本。
        """
        def tk(t): return self.model.tokenize(t)

        # 1. 区块 A: 音频之前的所有内容 (System + User Header)
        prefix_str = f"system\n{context or 'You are a helpful assistant.'}"
        prefix_tokens = [self.ID_IM_START] + tk(prefix_str) + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk("user\n") + [self.ID_AUDIO_START]

        # 2. 区块 B: 音频之后的所有内容 (Instruction + Assistant Header + History)
        suffix_head = f"assistant\n"
        if language: suffix_head += f"language {language}"

        suffix_tokens = [self.ID_AUDIO_END] + [self.ID_IM_END] + \
                        [self.ID_IM_START] + tk(suffix_head)
        if language:
            # 强制语言模式：<asr_text> 在 prompt 中，模型直接输出文本
            suffix_tokens += [self.ID_ASR_TEXT]
        suffix_tokens += tk(prefix_text)

        # 3. 统计并拼接
        n_pre, n_aud, n_suf = len(prefix_tokens), audio_embd.shape[0], len(suffix_tokens)
        total_embd = np.zeros((n_pre + n_aud + n_suf, self.model.n_embd), dtype=np.float32)

        total_embd[:n_pre] = self.embedding_table[prefix_tokens]
        total_embd[n_pre : n_pre + n_aud] = audio_embd
        total_embd[n_pre + n_aud:] = self.embedding_table[suffix_tokens]

        return total_embd

    @staticmethod
    def _parse_asr_output(raw: str, user_language: Optional[str] = None):
        """解析模型输出，提取 (language, text)

        user_language 非空时（强制模式），模型输出为纯文本，直接返回。
        user_language 为空时（自动检测），模型输出格式为
        "language X\\n<asr_text>transcription"，从中提取语言和文本。

        Returns:
            Tuple[str, str]: (language, text)，language 可能为空字符串表示未识别
        """
        if raw is None:
            return "", ""
        s = str(raw).strip()
        if not s:
            return "", ""

        if user_language:
            return user_language, s

        ASR_TEXT_TAG = "<asr_text>"
        if ASR_TEXT_TAG in s:
            meta_part, text_part = s.split(ASR_TEXT_TAG, 1)
        else:
            # 无 tag，整个输出视为文本
            return "", s.strip()

        lang = ""
        for line in meta_part.splitlines():
            line = line.strip()
            if line.lower().startswith("language "):
                val = line[len("language "):].strip()
                if val:
                    lang = normalize_language_name(val)
                break

        return lang, text_part.strip()

    def _decode(
        self, 
        full_embd: np.ndarray,
        prefix_text: str, 
        rollback_num: int,
        is_last_chunk: bool = False, 
        temperature: float = 0.4, 
        streaming: bool = True, 
    ) -> DecodeResult:
        """底层方法：执行单次 LLM 生成循环（物理推理）"""
        result = DecodeResult()
        
        total_len = full_embd.shape[0]
        pos_base = np.arange(0, total_len, dtype=np.int32)
        pos_arr = np.concatenate([pos_base, pos_base, pos_base, np.zeros(total_len, dtype=np.int32)])

        batch = llama.LlamaBatch(max(total_len * 4, 8192), self.model.n_embd, 1)
        batch.set_embd(full_embd, pos=pos_arr)

        # 1. Prefill
        # 诊断日志：prefill 前记录参数。GGML_ASSERT 越界走 C 层 abort()，会吞掉 print 缓冲，
        # 故用 logger（Unbuffered/StreamHandler 每次 emit 立即 flush，abort 吞不掉）。
        # pos_arr = total_len × 4（mrope 四平面），pos_arr_len > n_batch 即触发 get_rows 越界。
        _pos_arr_len = total_len * 4
        _overflow = _pos_arr_len > self.n_batch or total_len > self.n_ubatch
        logger.warning(
            f"[DIAG-ASR-PREFILL] total_len={total_len} pos_arr_len={_pos_arr_len} "
            f"n_batch={self.n_batch} n_ubatch={self.n_ubatch} n_ctx={self.n_ctx} "
            f"WILL_OVERFLOW={'YES' if _overflow else 'no'}"
        )
        self.ctx.clear_kv_cache()
        t_pre_start = time.time()
        self.ctx.decode(batch)
        prefill_time = time.time() - t_pre_start
        
        # 2. Generation Loop（使用新采样器和随机种子）
        t_gen_start = time.time()
        n_gen_tokens = 0
        display_queue = deque()
        stable_tokens = []
        stable_text_acc = ""
        text_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        
        # 诊断日志：生成阶段位置从 total_len 起逐 token 增长，最远到 total_len + 512，
        # 接近 n_ctx 才报警（正常 chunk 不刷屏）。
        if total_len + 512 > self.n_ctx:
            logger.error(
                f"[DIAG-ASR-GEN] start_pos={total_len} max_gen=512 farthest_pos={total_len + 512} "
                f"n_ctx={self.n_ctx} WILL_OVERFLOW_CTX=YES"
            )
        # 每次解码使用新的随机种子
        seed = int(np.random.randint(0, 2**31 - 1))
        sampler = llama.LlamaSampler(temperature=temperature, seed=seed)
        last_sampled_token = sampler.sample(self.ctx.ptr)
        for _ in range(512): # Max new tokens per chunk
            if last_sampled_token in [self.model.eos_token, self.ID_IM_END]:
                break
            
            if self.ctx.decode_token(last_sampled_token) != 0:
                    break
            
            display_queue.append(last_sampled_token)
            if len(display_queue) > rollback_num:
                ready_token = display_queue.popleft()
                stable_tokens.append(ready_token)
                piece = text_decoder.decode(self.model.token_to_bytes(ready_token))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end='', flush=True)
                    stable_text_acc += piece
            
            # 熔断检查：检测重复循环
            if len(stable_tokens) > 15:
                if len(set(stable_tokens[-15:])) <= 3:
                    result.is_aborted = True
                    break
            
            last_sampled_token = sampler.sample(self.ctx.ptr)
            n_gen_tokens += 1
            
        gen_time = time.time() - t_gen_start
        del sampler  # 释放采样器资源
        del batch
            
        if is_last_chunk and not result.is_aborted:
            while display_queue:
                t = display_queue.popleft()
                stable_tokens.append(t)
                piece = text_decoder.decode(self.model.token_to_bytes(t))
                if piece:
                    if streaming: print(re.sub(r'([，。？！：,\.])', r'\1\n', piece), end="", flush=True)
                    stable_text_acc += piece
            final_p = text_decoder.decode(b"", final=True)
            if final_p: 
                if streaming: print(final_p, end='', flush=True)
                stable_text_acc += final_p
        
        # 填充结果（内核输出标准化）
        result.text = stable_text_acc
        result.stable_tokens = stable_tokens
        result.t_prefill = prefill_time
        result.t_generate = gen_time
        result.n_prefill = total_len
        result.n_generate = n_gen_tokens
        result.n_generate = n_gen_tokens
        return result

    def _safe_decode(
        self, 
        full_embd: np.ndarray, 
        prefix_text: str, 
        rollback_num: int, 
        is_last_chunk: bool, 
        temperature: float, 
        streaming: bool = True, 
    ) -> DecodeResult:
        """带熔断加温重试的高层推理封装"""
        for i in range(4):
            res = self._decode(full_embd, prefix_text, rollback_num, is_last_chunk, temperature, streaming=streaming)
            if not res.is_aborted:
                break
            temperature += 0.3
            res.text += "====解码有误，强制熔断===="
            print(f"\n\n[!] 触发重试 (Temp -> {temperature:.1f})\n")
        return res 

    def _print_stats(self, stats: dict, audio_duration: float, t_total: float):
        """打印转录过程的性能统计指标"""
        rtf = t_total / audio_duration if audio_duration > 0 else 0
        pre_speed = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else 0
        gen_speed = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else 0
        
        print(f"\n\n📊 性能统计:")
        print(f"  🔹 RTF (实时率) : {rtf:.3f} (越小越快)")
        print(f"  🔹 音频时长    : {audio_duration:.2f} 秒")
        print(f"  🔹 总处理耗时  : {t_total:.2f} 秒")
        if stats.get("align_time"):
            print(f"  🔹 对齐耗时    : {stats['align_time']:.3f} 秒")
        print(f"  🔹 编码耗时    : {stats['encode_time']:.3f} 秒")
        print(f"  🔹 LLM 预填充  : {stats['prefill_time']:.3f} 秒 ({stats['prefill_tokens']} tokens, {pre_speed:.1f} tokens/s)")
        print(f"  🔹 LLM 生成    : {stats['decode_time']:.3f} 秒 ({stats['decode_tokens']} tokens, {gen_speed:.1f} tokens/s)")

    def transcribe(
        self, 
        audio_file: str, 
        language: Optional[str] = None, 
        context: Optional[str] = None, 
        start_second: float = 0.0,
        duration: float = 0.0,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (从文件加载音频)"""
        from .audio import load_audio
        audio = load_audio(audio_file, start_second=start_second if start_second > 0 else None, duration=duration if duration > 0 else None)
        
        return self.asr(
            audio=audio,
            context=context or "",
            language=language,
            chunk_size_sec=self.config.chunk_size,
            memory_chunks=self.config.memory_num,
            temperature=temperature,
            rollback_num=rollback_num
        )

    def asr(
        self,
        audio: np.ndarray,
        context: Optional[str],
        language: Optional[str],
        chunk_size_sec: float = 40.0,
        memory_chunks: int = 2,
        temperature: float = 0.4,
        rollback_num: int = 5
    ) -> TranscribeResult:
        """运行完整转录流水线 (三级流水线：i+1 预取, i 识别, i-1 对齐)

        语言检测策略：
        - language 有值时：全部 chunk 使用强制语言模式（prompt 含 "language X<asr_text>"）
        - language 为 None：首个 chunk 使用自动检测模式（prompt 不含 <asr_text>），
          从模型输出中解析语言；后续 chunk 切换到强制模式使用检测到的语言。
        """
        # 语言归一化与校验
        if language:
            language = normalize_language_name(language)
            validate_language(language)

        sr = 16000
        samples_per_chunk = int(chunk_size_sec * sr)
        total_len = len(audio)
        num_chunks = int(np.ceil(total_len / samples_per_chunk))
        total_duration = total_len / sr

        # 记忆管理 (预定义所有分片的物理边界)
        all_segments: List[ASRS_Segment] = [
            ASRS_Segment(
                idx=i,
                audio_start=i * chunk_size_sec,
                audio_end=min((i + 1) * chunk_size_sec, total_duration)
            ) for i in range(num_chunks)
        ]
        asr_memory = deque(maxlen=memory_chunks) # 存储 (embd, text)
        total_full_text = ""
        all_aligned_items: List[ForcedAlignItem] = []

        # 统计指标
        stats = {
            "prefill_time": 0.0, "decode_time": 0.0,
            "prefill_tokens": 0, "decode_tokens": 0,
            "encode_time": 0.0, "align_time": 0.0,
        }
        t_main_start = time.time()

        # 语言检测状态：初始为用户提供的 language（可能为 None）
        detected_language = language

        # --- 顺序同步处理循环 ---
        cancelled = False
        for i in range(num_chunks):
            # cancel 检查：每个 chunk 边界查外部信号（粗粒度，当前 chunk 不可中断）
            if self._cancel_event is not None and self._cancel_event.is_set():
                cancelled = True
                if self.verbose: print("\n[取消] 收到取消信号，提前结束（已完成 chunk 保留）")
                break
            # 1. 编码第 i 片段
            s, e = i * samples_per_chunk, min((i + 1) * samples_per_chunk, total_len)
            chunk_data = audio[s:e]
            if len(chunk_data) < samples_per_chunk:
                chunk_data = np.pad(chunk_data, (0, samples_per_chunk - len(chunk_data)))

            audio_feature, enc_time = self.encoder.encode(chunk_data)
            stats["encode_time"] += enc_time
            was_last = (i == num_chunks - 1)

            # 2. 识别第 i 片段文字
            # 首 chunk 自动检测，后续 chunk 使用已检测到的语言
            effective_language = detected_language
            prefix_text = "".join([m[1] for m in asr_memory])
            combined_audio = np.concatenate([m[0] for m in asr_memory] + [audio_feature], axis=0)
            full_embd = self._build_prompt_embd(combined_audio, prefix_text, context, effective_language)

            # 带熔断加温重试的解码调用
            res = self._safe_decode(full_embd, prefix_text, rollback_num, was_last, temperature)

            # 解析输出：自动检测模式下首 chunk 需提取语言
            chunk_lang, chunk_text = self._parse_asr_output(res.text, user_language=effective_language)
            if not detected_language and chunk_lang:
                detected_language = chunk_lang
                if self.verbose:
                    print(f"\n[语言检测] 自动检测到语言: {detected_language}")

            # 更新记忆与统计（仅存储纯文本，不含语言元数据）
            all_segments[i].text = chunk_text
            asr_memory.append((audio_feature, chunk_text))

            total_full_text += chunk_text
            stats["prefill_tokens"] += res.n_prefill; stats["prefill_time"] += res.t_prefill
            stats["decode_tokens"] += res.n_generate; stats["decode_time"] += res.t_generate

            # 3. 对齐第 i 片段 (同步)
            if self.aligner and chunk_text.strip():
                t_align_start = time.time()
                # 计算偏移（同步版本逻辑简化：直接使用片起点，不考虑前片动态边界）
                offset_sec = all_segments[i].audio_start
                s_smpl, e_smpl = int(offset_sec * sr), int(all_segments[i].audio_end * sr)
                audio_slice = audio[s_smpl:e_smpl]

                align_res = self.aligner.align(
                    audio_slice,
                    chunk_text,
                    language=detected_language,
                    offset_sec=float(offset_sec)
                )
                all_segments[i].items = align_res.items
                all_aligned_items.extend(align_res.items)
                stats["align_time"] += (time.time() - t_align_start)

        # 4. 结果整理
        all_aligned_items.sort(key=lambda x: x.start_time)
        t_total = time.time() - t_main_start
        if self.verbose: self._print_stats(stats, total_duration, t_total)

        return TranscribeResult(
            text=total_full_text,
            language=detected_language or "",
            alignment=ForcedAlignResult(items=all_aligned_items) if all_aligned_items else None,
            performance={**stats, "cancelled": True} if cancelled else stats
        )
