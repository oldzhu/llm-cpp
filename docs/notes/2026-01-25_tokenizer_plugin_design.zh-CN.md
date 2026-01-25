# 2026-01-25：分词器/嵌入插件化设计讨论

## 插件式分词器/嵌入设计（总结）
- 借鉴后端分派（CPU/NVIDIA/AMD）模式：定义分词器接口（抽象基类），如 `encode(text) -> tokens` 和 `decode(tokens) -> text`。
- 每种分词器（字节、GPT-2 BPE、SentencePiece）单独实现，均继承该接口。
- 增加注册表/工厂，支持运行时（配置或命令行参数）选择分词器。
- 嵌入层只依赖词表大小和 token 类型，不依赖分词器内部实现。
- 明确文档化每种分词器和嵌入的 shape/约定，并为 encode/decode 循环和嵌入查找添加测试。

## 优点
- 可动态切换分词器和嵌入，便于实验。
- 各变体可独立调试/测试。
- 未来可平滑扩展新分词方法。

## 下一步：设计草图（类/结构体布局与注册表模式）

---

## 设计草图：C++ 风格伪代码

```cpp
// 分词器接口
struct Tokenizer {
  virtual std::vector<int> encode(const std::string& text) const = 0;
  virtual std::string decode(const std::vector<int>& tokens) const = 0;
  virtual int vocab_size() const = 0;
  virtual ~Tokenizer() {}
};

// 字节级分词器
struct ByteTokenizer : public Tokenizer {
  // ...实现字节级 encode/decode...
};

// GPT-2 BPE 分词器
struct BpeTokenizer : public Tokenizer {
  // ...实现 BPE encode/decode...
};

// SentencePiece Unigram 分词器
struct SpUnigramTokenizer : public Tokenizer {
  // ...实现 SentencePiece encode/decode...
};

// 分词器注册表/工厂
struct TokenizerRegistry {
  static Tokenizer* create(const std::string& name, const std::string& vocab_path);
};

// 用法示例
std::unique_ptr<Tokenizer> tok(TokenizerRegistry::create("bpe", "bpe_vocab.txt"));
auto tokens = tok->encode("你好，世界！");
std::string text = tok->decode(tokens);
```

---

## 设计草图：抽象结构图

```
+-------------------+
|   Tokenizer (ABC) |
+-------------------+
        ^
        |
+-------------------+    +-------------------+    +--------------------------+
|   ByteTokenizer   |    |   BpeTokenizer    |    |   SpUnigramTokenizer     |
+-------------------+    +-------------------+    +--------------------------+

均实现：
  encode(text) -> tokens
  decode(tokens) -> text
  vocab_size()

+-------------------+
| TokenizerRegistry |
+-------------------+
      |
      v
+-------------------+
|   Tokenizer*      |
+-------------------+

嵌入层：
  - 只依赖 vocab_size() 和 token 类型
  - 不依赖分词器内部实现
```
