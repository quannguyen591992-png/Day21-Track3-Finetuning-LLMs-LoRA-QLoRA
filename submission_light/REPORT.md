# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Hải Quân — 2A202600660  
**Ngày nộp**: 2026-06-25  
**Submission option**: A — Lightweight ZIP  

---

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit + LoRA adapter
- **Frameworks/tools**: Unsloth, TRL `SFTTrainer`, PEFT, bitsandbytes, Transformers, datasets
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`
- **Dataset size used**: 200 samples
  - Train: 180 samples
  - Eval: 20 samples
- **Dataset format**: Alpaca-style text format with `Instruction`, optional `Input`, and `Response`
- **GPU**: Tesla T4 on Google Colab, approximately 15 GB VRAM
- **LoRA target modules**: `q_proj`, `v_proj`
- **Baseline LoRA config**:
  - `r = 16`
  - `lora_alpha = 32`
  - `lora_dropout = 0`
  - `bias = "none"`
  - gradient checkpointing enabled through Unsloth
- **Training setup**:
  - 3 epochs
  - learning rate: `2e-4`
  - scheduler: cosine
  - warmup ratio: `0.10`
  - train batch size: 1
  - gradient accumulation steps: 8
  - effective batch size: 8
  - optimizer: `adamw_8bit`
- **Total measured training time for 3 ranks**: approximately 12.37 minutes
- **Estimated training cost**: approximately `$0.07` assuming T4 cost around `$0.35/hour`

---

## 2. Rank Experiment Results

The experiment trained three LoRA adapters with different ranks while keeping the base model, dataset, and other hyperparameters the same. The only changed variables were LoRA rank `r` and `lora_alpha`.

| Rank | Alpha | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 1,843,200 | 4.05 | 7.22 | 1.5577 | 4.7479 |
| 16 | 32 | 3,686,400 | 4.29 | 6.62 | 1.5161 | 4.5544 |
| 64 | 128 | 14,745,600 | 4.03 | 8.00 | 1.4768 | 4.3790 |

### Observations

- Increasing rank improved eval loss and perplexity in this run.
- `r=8` used the fewest trainable parameters, but it had the highest perplexity: **4.7479**.
- `r=16` improved perplexity to **4.5544**, while still keeping the trainable parameter count small.
- `r=64` achieved the best perplexity: **4.3790**, but required 4x more trainable parameters than `r=16`.
- Peak VRAM increased from around **6.62–7.22 GB** to about **8.00 GB** for `r=64`, which is still manageable on a T4 but less lightweight.

Although `r=64` achieved the best quantitative result, the gain from `r=16` to `r=64` is relatively modest compared with the increase in trainable parameters. This shows the expected LoRA rank trade-off: higher rank gives more adaptation capacity, but the improvement may have diminishing returns.

---

## 3. Loss Curve Analysis

In the T4 notebook configuration, evaluation during training was disabled to reduce VRAM usage. Therefore, the loss curve mainly reflects training loss rather than both training and validation loss.

From the completed run, all three adapters trained successfully and produced valid eval loss/perplexity values after training. The final eval losses were:

- `r=8`: 1.5577
- `r=16`: 1.5161
- `r=64`: 1.4768

The decreasing eval loss as rank increases suggests that the model benefited from additional LoRA capacity on this dataset. However, because the eval set contains only 20 samples, the differences should be interpreted carefully. A larger validation set and more qualitative prompts would give a more reliable estimate of generalization.

There is no clear evidence of severe overfitting from the final eval numbers alone. However, because this lab used only 200 samples, overfitting remains a possible risk, especially with higher-rank adapters such as `r=64`. If I extended this experiment, I would enable more frequent evaluation on a larger validation set or track training/eval loss using W&B to better detect overfitting.

---

## 4. Qualitative Comparison

The qualitative comparison used five Vietnamese prompts and compared the base model response with the fine-tuned `r=16` adapter response.

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base model**:

> Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu...

**Fine-tuned model (`r=16`)**:

> Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI...

**Nhận xét**: Fine-tuned output có phong cách giải thích rõ hơn một chút và phù hợp với câu hỏi dành cho người mới bắt đầu. Cả hai câu trả lời đều đúng hướng, nhưng câu trả lời fine-tuned có cấu trúc tự nhiên hơn.

---

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base model**:

> Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
>
> ```python
> def fibonacci(n):
>     if n <= 0:
>         return "N phải là một số dương"
>     elif n == 1:
>         return 0
>     elif n == 2:
>         return 1
> ```

**Fine-tuned model (`r=16`)**:

> Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
>
> ```python
> def fibonacci(n):
>     if n < 0:
>         raise ValueError("Input phải là một số nguyên dương.")
>     elif n == 0:
>         return 0
>     elif n == 1:
>         return 1
>     else:
>         a, b = 0, 1
>         for _ in range...
> ```

**Nhận xét**: Fine-tuned output tốt hơn vì dùng hướng iterative rõ ràng hơn, xử lý case `n < 0`, `n == 0`, `n == 1`. Câu trả lời base model bị cắt và logic indexing có thể gây nhầm lẫn.

---

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base model**:

> 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện...

**Fine-tuned model (`r=16`)**:

> 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản...

**Nhận xét**: Fine-tuned output có dạng liệt kê đúng yêu cầu hơn. Tuy nhiên một số thuật ngữ như “Chuyển đổi” và “Tương thích” còn hơi chung chung. Base model giải thích dài hơn nhưng bị lặp ý “thân thiện”.

---

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base model**:

> LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp...

**Fine-tuned model (`r=16`)**:

> LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định...

**Nhận xét**: Đây là một case degraded. Fine-tuned model đã giải thích sai chữ viết tắt LoRA: đúng là **Low-Rank Adaptation**, không phải “Layer-wise Adaptive Regularization Optimization”. Điều này cho thấy fine-tuning trên dataset chung không đảm bảo cải thiện kiến thức chuyên môn nếu dataset không chứa đúng kiến thức domain. Với kiến thức factual, RAG hoặc dataset domain-specific tốt hơn.

---

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base model**:

> Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học...

**Fine-tuned model (`r=16`)**:

> Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh...

**Nhận xét**: Fine-tuned model trả lời tự nhiên hơn, nhưng vẫn còn khá tổng quát. Cả hai câu trả lời chưa phân biệt thật sắc nét giữa ba phương pháp. Đây là ví dụ cho thấy fine-tuning với dataset general Vietnamese Alpaca chủ yếu giúp style/format, chưa chắc cải thiện reasoning hoặc kiến thức chuyên sâu.

---

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, rank cao hơn giúp cải thiện eval loss và perplexity. `r=8` là cấu hình nhẹ nhất với 1.84M trainable parameters, nhưng có perplexity cao nhất là 4.7479. `r=16` tăng số trainable parameters lên 3.69M và cải thiện perplexity xuống 4.5544. `r=64` đạt kết quả tốt nhất với perplexity 4.3790, nhưng cần 14.75M trainable parameters, tức gấp 4 lần `r=16` và gấp 8 lần `r=8`.

Nếu chỉ nhìn vào perplexity, `r=64` là adapter tốt nhất trong thí nghiệm này. Tuy nhiên, nếu xét ROI cho production, `r=16` là lựa chọn cân bằng hơn vì chất lượng gần với `r=64` nhưng nhẹ hơn nhiều. Việc tăng từ `r=16` lên `r=64` làm tăng đáng kể số parameter cần train và lưu trữ, nhưng mức cải thiện perplexity không quá lớn. Điều này thể hiện diminishing returns: rank càng cao thì model có nhiều capacity hơn, nhưng lợi ích bổ sung không tăng tuyến tính theo số parameter.

Recommendation của tôi là dùng `r=16` cho bài toán production nhỏ hoặc vừa, đặc biệt khi cần tiết kiệm VRAM, storage và chi phí training. `r=64` phù hợp hơn khi dataset lớn hơn, domain phức tạp hơn, hoặc khi chất lượng là ưu tiên cao hơn chi phí. Với dataset nhỏ 200 samples trong lab này, `r=16` là lựa chọn hợp lý nhất để cân bằng giữa hiệu quả và tài nguyên.

---

## 6. What I Learned

- LoRA cho phép fine-tune LLM bằng cách chỉ train một lượng nhỏ parameter thay vì cập nhật toàn bộ base model. Điều này giúp giảm rất nhiều chi phí training và VRAM.
- QLoRA giúp load base model ở 4-bit, nhờ đó có thể fine-tune model vài tỷ tham số trên GPU T4 miễn phí của Colab.
- Rank là một hyperparameter quan trọng. Rank thấp giúp adapter nhẹ và nhanh, rank cao tăng capacity nhưng có thể gặp diminishing returns.
- Perplexity là chỉ số hữu ích để so sánh định lượng, nhưng không đủ. Cần qualitative evaluation vì model có thể có perplexity tốt hơn nhưng vẫn trả lời sai kiến thức factual.
- Fine-tuning phù hợp để học style, format và hành vi trả lời. Với kiến thức mới hoặc factual knowledge, RAG thường phù hợp hơn fine-tuning.

---

## Submission Files

Current lightweight submission folder contains:

```text
submission_light/
├── REPORT.md
├── adapters/
│   └── r16/
│       ├── adapter_config.json
│       └── adapter_model.safetensors
└── results/
    ├── qualitative_comparison.csv
    └── rank_experiment_summary.csv
```
