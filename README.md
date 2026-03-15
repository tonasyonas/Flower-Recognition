# SC4001: Neural Networks and Deep Learning - Group Project

## Track F: Advanced Few-Shot Flower Recognition

**Team Members:**
- Jonas Tua 
- Jia Ying Thor
- F Syed Abu Thahir

### Proposed Technique & Architecture
- **Base Model:** Pre-trained Vision Transformer (ViT).
- **Visual Prompt Tuning (VPT):** Freezing the pre-trained transformer backbone and learning a small set of prompt parameters.
- **Advanced Augmentation:** MixUp.
- **Loss Function:** Triplet Loss alongside Cross-Entropy.
