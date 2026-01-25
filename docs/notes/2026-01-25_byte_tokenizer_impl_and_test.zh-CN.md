# 2026-01-25：ByteTokenizer 实现与测试结果

## 实现
- 新增 `Tokenizer` 抽象基类（`src/tokenizer/tokenizer.h`）。
- 实现 `ByteTokenizer`（`src/tokenizer/byte_tokenizer.h/cpp`）：
  - `encode(const std::string&) -> std::vector<int>`：1 字节 = 1 token。
  - `decode(const std::vector<int>&) -> std::string`：从 token 还原字符串。
  - `vocab_size() == 256`。
- 集成到 CLI/模型：prompt 编码和模型配置均走 tokenizer 接口。
- 更新 CMakeLists.txt，包含新源码。

## 测试
- 增加 encode/decode 循环测试（ASCII、UTF-8/中文、空串、全字节）。
- 验证 `vocab_size == 256`。
- 检查 embedding shape 与词表大小一致。
- 全部测试通过（`ctest`）：ByteTokenizer 正确可用。
