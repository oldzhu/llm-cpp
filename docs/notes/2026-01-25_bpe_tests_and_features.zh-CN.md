# 2026-01-25：BPE 分词器——测试与功能建议

## 推荐 BPE 测试
- 各类字符串的 encode/decode 循环（ASCII、Unicode、边界情况）
- 多 token 合并测试（如 "abcde" → ["abc", "de"]）
- 未知/非法输入处理（应抛异常或兜底）
- 词表大小、token-to-id/id-to-token 一致性
- 用真实 GPT-2 词表/合并文件测试（如有）
- CLI 集成测试：用 --tokenizer bpe 跑通并检查输出

## 可扩展 BPE 功能
- 字节兜底：对词表外字节编码为特殊 token
- Unicode 感知切分：更健壮地处理多字节 UTF-8 字符
- BPE 合并缓存，加速编码
- 支持特殊 token（如 <unk>、<pad>、<bos>、<eos>）
- 词表/合并文件支持 JSON 或其他格式导入导出
- 支持 BPE 训练（从数据学习合并规则，而非仅加载）
- 支持 SentencePiece Unigram 或 WordPiece 等分词器
- CLI 增加字符串分词结果打印选项
