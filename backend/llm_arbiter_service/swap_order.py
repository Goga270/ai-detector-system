class SwapOrderManager:
    def __init__(self, target_agent):
        self.agent = target_agent

    async def balanced_analysis(self, text: str, bert: float, dgpt: float):
        order_1 = f"ОЦЕНИ ТЕКСТ: {text}\nМЕТРИКИ: BERT={bert}, D-GPT={dgpt}"
        res_1 = await self.agent.client.generate(self.agent.system_prompt, order_1)

        order_2 = f"МЕТРИКИ: BERT={bert}, D-GPT={dgpt}\nОЦЕНИ ТЕКСТ: {text}"
        res_2 = await self.agent.client.generate(self.agent.system_prompt, order_2)

        return res_1, res_2