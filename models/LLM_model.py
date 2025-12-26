# 获取日志记录器
import json
import logging
import os
import time
from abc import ABC, abstractmethod

import backoff
from openai import OpenAI
from google import genai


from tools.logConfig import ERROR_ICON, SUCCESS_ICON, WAIT_ICON

logger = logging.getLogger('debate_room')


class LLMClient(ABC):
    """LLM 客户端抽象基类"""

    @abstractmethod
    def get_completion(self, messages, **kwargs):
        """获取模型回答"""
        pass

class GeminiClient(LLMClient):
    """Google Gemini API 客户端"""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

        if not self.api_key:
            logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables")

        # 初始化 Gemini 客户端
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"{SUCCESS_ICON} Gemini 客户端初始化成功")

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300,
        giveup=lambda e: "AFC is enabled" not in str(e)
    )
    def generate_content_with_retry(self, contents, config=None):
        """带重试机制的内容生成函数"""
        try:
            logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
            logger.debug(f"请求内容: {contents}")
            logger.debug(f"请求配置: {config}")

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            logger.info(f"{SUCCESS_ICON} API 调用成功")
            logger.debug(f"响应内容: {response.text[:500]}...")
            return response
        except Exception as e:
            error_msg = str(e)
            if "location" in error_msg.lower():
                logger.info(
                    f"\033[91m❗ Gemini API 地理位置限制错误: 请使用美国节点VPN后重试\033[0m")
                logger.error(f"详细错误: {error_msg}")
            elif "AFC is enabled" in error_msg:
                logger.warning(
                    f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {error_msg}")
                time.sleep(5)
            else:
                logger.error(f"{ERROR_ICON} API 调用失败: {error_msg}")
            raise e

    def get_completion(self, messages, max_retries=3, initial_retry_delay=1, **kwargs):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            logger.info(f"{WAIT_ICON} 使用 Gemini 模型: {self.model}")
            logger.debug(f"消息内容: {messages}")

            for attempt in range(max_retries):
                try:
                    # 转换消息格式
                    prompt = ""
                    system_instruction = None

                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        if role == "system":
                            system_instruction = content
                        elif role == "user":
                            prompt += f"User: {content}\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n"

                    # 准备配置
                    config = {}
                    if system_instruction:
                        config['system_instruction'] = system_instruction

                    # 调用 API
                    response = self.generate_content_with_retry(
                        contents=prompt.strip(),
                        config=config
                    )

                    if response is None:
                        logger.warning(
                            f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            logger.info(
                                f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    logger.debug(f"API 原始响应: {response.text}")
                    logger.info(f"{SUCCESS_ICON} 成功获取 Gemini 响应")

                    # 直接返回文本内容
                    return response.text

                except Exception as e:
                    logger.error(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                        return None

        except Exception as e:
            logger.error(f"{ERROR_ICON} get_completion 发生错误: {str(e)}")
            return None


def LLM_process(bull,  bear):
    bull_text = bull.text
    bear_text = bear.text
    logger.info(
        f"已获取看多观点(置信度: {bull_text.get('confidence', 0)})和看空观点(置信度: {bear_text.get('confidence', 0)})")

    # 比较置信度级别
    bull_confidence = bull_text.get("confidence", 0)
    bear_confidence = bear_text.get("confidence", 0)

    # 分析辩论观点
    debate_summary = []
    debate_summary.append("Bullish Arguments:")
    for point in bull_text.get("thesis_points", []):
        debate_summary.append(f"+ {point}")

    debate_summary.append("\nBearish Arguments:")
    for point in bear_text.get("thesis_points", []):
        debate_summary.append(f"- {point}")

    # 收集所有研究员的论点，准备发给 LLM
    all_perspectives = {}
    # for name, data in researcher_data.items():
    #     perspective = data.get("perspective", name.replace(
    #         "researcher_", "").replace("_agent", ""))
    #     all_perspectives[perspective] = {
    #         "confidence": data.get("confidence", 0),
    #         "thesis_points": data.get("thesis_points", [])
    #     }

    logger.info(f"准备让 LLM 分析 {len(all_perspectives)} 个研究员的观点")
    # 构建发送给 LLM 的提示
    llm_prompt = """
    你是一位专业的金融分析师，请分析以下投资研究员的观点，并给出你的第三方分析:

    """
    for perspective, data in all_perspectives.items():
        llm_prompt += f"\n{perspective.upper()} 观点 (置信度: {data['confidence']}):\n"
        for point in data["thesis_points"]:
            llm_prompt += f"- {point}\n"

    llm_prompt += """
    请提供以下格式的 JSON 回复:
    {
        "analysis": "你的详细分析，评估各方观点的优劣，并指出你认为最有说服力的论点",
        "score": 0.5,  // 你的评分，从 -1.0(极度看空) 到 1.0(极度看多)，0 表示中性
        "reasoning": "你给出这个评分的简要理由"
    }

    务必确保你的回复是有效的 JSON 格式，且包含上述所有字段。回复必须使用英文，不要使用中文或其他语言。
    """
    # 调用 LLM 获取第三方观点
    llm_response = None
    llm_analysis = None
    llm_score = 0  # 默认为中性
    messages = [
        {"role": "system",
         "content": "You are a professional financial analyst. Please provide your analysis in English only, not in Chinese or any other language."},
        {"role": "user", "content": llm_prompt}
    ]

    llm_response = llm_chat(messages=messages, max_retry=3)
    if llm_response:
        try:
            # 尝试提取 JSON 部分
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
                llm_analysis = json.loads(json_str)
                llm_score = float(llm_analysis.get("score", 0))
                # 确保分数在有效范围内
                llm_score = max(min(llm_score, 1.0), -1.0)
                logger.info(f"成功解析 LLM 回复，评分: {llm_score}")
                logger.debug(
                    f"LLM 分析内容: {llm_analysis.get('analysis', '未提供分析')[:100]}...")
        except Exception as e:
            # 如果解析失败，记录错误并使用默认值
            logger.error(f"解析 LLM 回复失败: {e}")
            llm_analysis = {"analysis": "Failed to parse LLM response",
                            "score": 0, "reasoning": "Parsing error"}




def llm_chat(messages, model=None, max_retry=3, client_type="gemini", api_key=None, base_url=None):
    if client_type == "auto":
        if(api_key and  base_url and  model):
            client_typ = "openai_compatible"
        else:
            client_type = "gemini"
    if client_type == "gemini":
        client = GeminiClient(
            api_key=api_key,
            model=model
        )
    return client.get_completion(messages)