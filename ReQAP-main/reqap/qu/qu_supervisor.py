from __future__ import annotations

from typing import Dict, List

# NOTE:
# 在 Windows 环境下 vllm 可能无法安装成功。
# QUSupervisor 在 seq2seq 模式下不会真正使用 InstructModel/OpenAIModel，
# 但如果在这里进行运行期 import，会导致 import 阶段就崩溃。
# 因此将这些 import 注释掉，仅保留类型标注的字符串化（由 __future__ 控制）。
# from reqap.llm.instruct_model import InstructModel
# from reqap.llm.openai import OpenAIModel


class QUSupervisor:
    def __init__(self, icl_model: InstructModel | OpenAIModel):
        self.icl_model = icl_model
    
    def inference(self, dialog: List[Dict], sampling_params: Dict={}) -> str:
        return self.icl_model.inference(dialog, sampling_params)
    
    def batch_inference(self, dialogs: List[List[Dict]], sampling_params: Dict={}) -> List[str]:
        return self.icl_model.batch_inference(dialogs, sampling_params)
