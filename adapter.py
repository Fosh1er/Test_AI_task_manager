from langchain_core.runnables import RunnableLambda
from provider import Provider
from logger import logger

def create_langchain_adapter(provider: Provider):
    def _invoke(input):
        # LangChain prompts pass a ChatPromptValue, not a dict
        if hasattr(input, "to_messages"):
            messages = input.to_messages()
        elif isinstance(input, dict):
            messages = input.get("messages", [])
        else:
            logger.error(f"Adapter received unexpected input type: {type(input)}")
            return ""

        if not messages:
            logger.error("Adapter received empty messages")
            return ""

        system_text = ""
        for msg in messages:
            if msg.type == "system":
                system_text = msg.content
                break
        user_text = messages[-1].content

        # Вызываем провайдер
        text = provider.complete(system_text, user_text, temperature=0)

        # После вызова берём метрики из провайдера
        info = {}
        try:
            # Доступ к внутреннему клиенту через provider.client
            client = provider.client
            if hasattr(client, 'last_response_info'):
                info = client.last_response_info
        except Exception as e:
            logger.debug(f"Could not get response info: {e}")

        if info:
            tps = info.get("tokens_per_sec")
            if tps:
                logger.info(f"Speed: {tps:.1f} tok/s | Tokens: {info.get('tokens')} | Time: {info['time_seconds']:.2f}s")
            else:
                logger.info(f"Response time: {info['time_seconds']:.2f}s")
        return text
    return RunnableLambda(_invoke)