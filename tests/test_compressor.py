"""上下文压缩单元测试。"""
from services.context_compressor import ContextCompressor


class TestHeuristicCompression:
    def setup_method(self):
        self.compressor = ContextCompressor(llm_service=None)

    def test_short_chunk_kept(self):
        chunk = "短文本。"
        out = self.compressor.compress("问题", [chunk])
        assert out == [chunk]

    def test_long_chunk_reduced(self):
        chunk = (
            "OpenAI 是一家人工智能公司。OpenAI 发布了 ChatGPT 产品。"
            "ChatGPT 基于 GPT 架构。量子计算使用量子比特。火星是行星。"
        )
        out = self.compressor.compress("OpenAI ChatGPT 发布了什么", [chunk])
        # 压缩后保留含查询词的句子
        assert "OpenAI" in out[0]
        assert "火星" not in out[0]

    def test_query_terms(self):
        terms = ContextCompressor._query_terms("OpenAI ChatGPT")
        assert "openai" in terms and "chatgpt" in terms

    def test_split_sentences(self):
        sents = ContextCompressor._split_sentences("第一句。第二句！第三句")
        assert len(sents) == 3
