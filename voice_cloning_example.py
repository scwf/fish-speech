#!/usr/bin/env python3
"""
Fish Speech 语音克隆示例代码
基于 Fish Speech 的语音克隆功能实现

使用前请确保：
1. 已安装 fish-speech: pip install -e .
2. 下载模型: 使用 tools/download_models.py 自动下载模型，命令如下：
   ```bash
   python tools/download_models.py
   ```
3. 已准备参考音频和对应文本
"""

import torch
import os
from pathlib import Path
from typing import List, Optional, Union

# 方法1: 直接调用推理引擎
class DirectVoiceCloner:
    """直接调用推理引擎进行语音克隆"""
    
    def __init__(self, 
                 llama_checkpoint_path: str = "checkpoints/openaudio-s1-mini",
                 decoder_checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
                 device: str = "cuda",
                 precision: torch.dtype = torch.bfloat16,
                 compile: bool = True):
        
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.models.dac.inference import load_model as load_decoder_model
        except ImportError:
            raise ImportError("请先安装 fish-speech: pip install -e .")
        
        # 初始化LLM队列
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=device,
            precision=precision,
            compile=compile,
        )
        
        # 初始化解码器
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_checkpoint_path,
            device=device,
        )
        
        # 创建推理引擎
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=precision,
            compile=compile,
        )
    
    def clone_voice(self,
                    text: str,
                    reference_audio_path: str,
                    reference_text: str,
                    output_path: str = "cloned_audio.wav",
                    **kwargs) -> bool:
        """
        单条语音克隆
        
        参数:
            text: 要合成的文本内容
            reference_audio_path: 参考音频文件路径
            reference_text: 与参考音频对应的文本内容
            output_path: 输出音频文件路径
            **kwargs: 其他参数，如temperature, top_p等
        """
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
        import scipy.io.wavfile as wavfile
        
        # 读取参考音频
        try:
            with open(reference_audio_path, "rb") as f:
                reference_audio_bytes = f.read()
        except FileNotFoundError:
            print(f"参考音频文件未找到: {reference_audio_path}")
            return False
        
        # 构建请求
        request = ServeTTSRequest(
            text=text,
            references=[
                ServeReferenceAudio(
                    audio=reference_audio_bytes,
                    text=reference_text
                )
            ],
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            chunk_length=kwargs.get("chunk_length", 100),
            top_p=kwargs.get("top_p", 0.6),
            repetition_penalty=kwargs.get("repetition_penalty", 1.1),
            temperature=kwargs.get("temperature", 0.2),
            format=kwargs.get("format", "wav"),
            seed=kwargs.get("seed", 42),
            use_memory_cache=kwargs.get("use_memory_cache", "on")
        )
        
        # 执行推理
        try:
            results = list(self.engine.inference(request))
            final_result = results[-1]
            
            if final_result.code == "final":
                sample_rate, audio_data = final_result.audio
                wavfile.write(output_path, sample_rate, audio_data)
                print(f"语音克隆完成: {output_path}")
                return True
            else:
                print(f"语音克隆失败: {final_result.error}")
                return False
                
        except Exception as e:
            print(f"推理过程中出错: {e}")
            return False

# 方法2: 批量语音克隆（使用直接推理引擎）
class BatchVoiceCloner:
    """批量语音克隆（使用直接推理引擎）"""
    
    def __init__(self, 
                 llama_checkpoint_path: str = "checkpoints/openaudio-s1-mini",
                 decoder_checkpoint_path: str = "checkpoints/openaudio-s1-mini/codec.pth",
                 device: str = "cuda",
                 precision: torch.dtype = torch.bfloat16,
                 compile: bool = True):
        
        try:
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.models.dac.inference import load_model as load_decoder_model
        except ImportError:
            raise ImportError("请先安装 fish-speech: pip install -e .")
        
        # 初始化LLM队列
        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=device,
            precision=precision,
            compile=compile,
        )
        
        # 初始化解码器
        self.decoder_model = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=decoder_checkpoint_path,
            device=device,
        )
        
        # 创建推理引擎
        self.engine = TTSInferenceEngine(
            llama_queue=self.llama_queue,
            decoder_model=self.decoder_model,
            precision=precision,
            compile=compile,
        )
    
    def clone_voices(self,
                    texts: List[str],
                    reference_audio_path: str,
                    reference_text: str,
                    output_dir: str = "./batch_output",
                    **kwargs) -> List[str]:
        """
        批量语音克隆
        
        参数:
            texts: 要合成的文本列表
            reference_audio_path: 参考音频文件路径
            reference_text: 与参考音频对应的文本内容
            output_dir: 输出目录
            **kwargs: 其他参数，如temperature, top_p等
        
        返回:
            成功生成的文件路径列表
        """
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
        import scipy.io.wavfile as wavfile
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取参考音频
        try:
            with open(reference_audio_path, "rb") as f:
                reference_audio_bytes = f.read()
        except FileNotFoundError:
            print(f"参考音频文件未找到: {reference_audio_path}")
            return []
        
        successful_results = []
        
        for i, text in enumerate(texts):
            try:
                # 构建请求
                request = ServeTTSRequest(
                    text=text,
                    references=[
                        ServeReferenceAudio(
                            audio=reference_audio_bytes,
                            text=reference_text
                        )
                    ],
                    max_new_tokens=kwargs.get("max_new_tokens", 1024),
                    chunk_length=kwargs.get("chunk_length", 100),
                    top_p=kwargs.get("top_p", 0.6),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                    temperature=kwargs.get("temperature", 0.2),
                    format=kwargs.get("format", "wav"),
                    seed=kwargs.get("seed", 42),
                    use_memory_cache=kwargs.get("use_memory_cache", "on")
                )
                
                # 执行推理
                results = list(self.engine.inference(request))
                final_result = results[-1]
                
                if final_result.code == "final":
                    sample_rate, audio_data = final_result.audio
                    output_path = os.path.join(output_dir, f"cloned_{i}.wav")
                    wavfile.write(output_path, sample_rate, audio_data)
                    print(f"已完成: {output_path}")
                    successful_results.append(output_path)
                else:
                    print(f"第{i}个语音克隆失败: {final_result.error}")
                    
            except Exception as e:
                print(f"处理第{i}个文本时出错: {e}")
        
        print(f"批量处理完成: {len(successful_results)}/{len(texts)} 成功")
        return successful_results

# 语音克隆质量优化工具
class VoiceCloningOptimizer:
    """语音克隆质量优化"""
    
    @staticmethod
    def get_high_fidelity_params() -> dict:
        """获取高质量语音克隆参数"""
        return {
            "temperature": 0.2,
            "top_p": 0.6,
            "repetition_penalty": 1.4,
            "chunk_length": 100,
            "max_new_tokens": 800
        }
    
    @staticmethod
    def get_balanced_params() -> dict:
        """获取平衡参数"""
        return {
            "temperature": 0.5,
            "top_p": 0.7,
            "repetition_penalty": 1.1,
            "chunk_length": 150,
            "max_new_tokens": 1000
        }
    
    @staticmethod
    def get_diverse_params() -> dict:
        """获取多样性参数"""
        return {
            "temperature": 0.8,
            "top_p": 0.8,
            "repetition_penalty": 1.0,
            "chunk_length": 200,
            "max_new_tokens": 1024
        }

# 使用示例和测试函数
def main():
    """使用示例"""
    
    # 用户需要填写的参数
    REFERENCE_AUDIO_PATH = "/home/xiaofei/code/index-tts/refer_voice/qjc_short.wav"  # 参考音频文件路径，例如: "my_voice.wav"
    REFERENCE_TEXT = "你说遇事不决，可问春风，春风不语，即随本心"        # 与参考音频对应的文本内容
    OUTPUT_TEXT = """
        遇事不决，可问春风。
        君子坐而论道，少年起而行之。
        世间万般讲理与不讲理，终归会落在一处，我心安处即吾乡。
        少年的肩膀，就该这样才对嘛，什么家国仇恨，浩然正气的，都不要急，先挑起清风明月、杨柳依依和草长莺飞。
        阿良左右，一竖一横，剑道剑术，共斩蛮荒。
        山外风雨三尺剑，有事提剑下山去。
    """
    
    # 示例：使用直接推理引擎
    print("=== 直接推理引擎示例 ===")
    if REFERENCE_AUDIO_PATH and REFERENCE_TEXT and OUTPUT_TEXT:
        cloner = DirectVoiceCloner()
        success = cloner.clone_voice(
            text=OUTPUT_TEXT,
            reference_audio_path=REFERENCE_AUDIO_PATH,
            reference_text=REFERENCE_TEXT,
            output_path="direct_output.wav",
            **VoiceCloningOptimizer.get_high_fidelity_params()
        )
    else:
        print("请填写 REFERENCE_AUDIO_PATH, REFERENCE_TEXT 和 OUTPUT_TEXT")
    
    # 示例：批量语音克隆
    print("\n=== 批量克隆示例 ===")
    if REFERENCE_AUDIO_PATH and REFERENCE_TEXT:
        batch_texts = [
            "遇事不决，可问春风。",
            "君子坐而论道，少年起而行之。",
            "世间万般讲理与不讲理，终归会落在一处，我心安处即吾乡。",
            "少年的肩膀，就该这样才对嘛，什么家国仇恨，浩然正气的，都不要急，先挑起清风明月、杨柳依依和草长莺飞。",
            "阿良左右，一竖一横，剑道剑术，共斩蛮荒。",
            "山外风雨三尺剑，有事提剑下山去。"
        ]
        
        batch_cloner = BatchVoiceCloner()
        results = batch_cloner.clone_voices(
            texts=batch_texts,
            reference_audio_path=REFERENCE_AUDIO_PATH,
            reference_text=REFERENCE_TEXT,
            output_dir="./batch_results",
            **VoiceCloningOptimizer.get_high_fidelity_params()
        )
    else:
        print("请填写 REFERENCE_AUDIO_PATH 和 REFERENCE_TEXT")

if __name__ == "__main__":
    main()