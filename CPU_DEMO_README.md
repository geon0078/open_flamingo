# OpenFlamingo CPU-Optimized Demo

This fork contains CPU-optimized versions of OpenFlamingo with interactive interfaces for low-memory environments.

## 🚀 New Features

### 1. CPU-Optimized Main Demo (`flamingo_demo.py`)
- **CPU-only mode** with memory optimization
- **FP16 support** for reduced memory usage
- **Thread limiting** for better CPU performance
- **Image resizing** to reduce memory footprint
- **Batch processing** limitations for stability

### 2. Interactive CLI Interface (`image_qa_interface.py`)
- **User-friendly command-line interface**
- **File upload support** for local images
- **Conversation-style Q&A**
- **Session management** with memory cleanup

### 3. Simple Q&A Tool (`simple_image_qa.py`)
- **Drag-and-drop style** image input
- **Quick question-answer** workflow
- **Minimal setup** required

### 4. Test Script (`simple_test.py`)
- **Quick validation** of setup
- **Basic functionality testing**
- **Error debugging** helpers

## 📋 Requirements

```
torch
torchvision
Pillow
requests
transformers
huggingface_hub
open_flamingo
```

## 🛠️ Installation

1. Install the base OpenFlamingo package:
```bash
pip install open-flamingo
```

2. Install additional requirements:
```bash
pip install -r demo_requirements.txt
```

## 🎯 Usage

### Basic Demo
```bash
python flamingo_demo.py
```

### Interactive Interface
```bash
python image_qa_interface.py
```

### Simple Q&A
```bash
python simple_image_qa.py
```

### Quick Test
```bash
python simple_test.py
```

## 💡 CPU Optimization Features

- **Forced CPU mode**: All computations run on CPU
- **Memory management**: Automatic garbage collection
- **FP16 precision**: Reduced memory usage
- **Image resizing**: Limited to 224x224 for efficiency
- **Thread control**: Optimized CPU thread usage
- **Batch limitations**: Prevents memory overflow

## ⚠️ Important Notes

1. **CPU mode is slower** than GPU but more accessible
2. **Memory usage** is optimized but still requires sufficient RAM
3. **Model loading** may take time on first run
4. **Image quality** is limited to 224x224 for memory efficiency

## 🔧 Troubleshooting

### Memory Issues
- Restart the system to free up memory
- Close other applications
- Use smaller models (3B instead of 9B)

### Performance Issues
- CPU mode is inherently slower
- Consider using fewer example images in few-shot learning
- Reduce max_tokens for faster generation

## 📝 Changes from Original

- Added CPU-only enforcement
- Implemented memory optimization strategies
- Created user-friendly interfaces
- Added error handling and recovery
- Included Korean language support in comments

## 🤝 Contributing

This is a fork focused on CPU optimization and accessibility. Feel free to submit issues or pull requests for improvements.

## 📄 License

Same as the original OpenFlamingo project.

---

**Author**: GEON AN (geon0078@g.eulji.ac.kr)  
**Based on**: OpenFlamingo by ML Foundations
