#!/usr/bin/env python3
"""
两阶段分类器推理脚本
用于实际使用和验证两阶段PhenoSSP分类器
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, classification_report, confusion_matrix
import h5py
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# --- 添加项目路径 ---
project_root = Path("/export/home/zhangyujia/PhenoSSP_project/PhenoSSP-main")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tutorials" / "utils"))
from cell_phenotyping import CellPhenotyping

# --- 设置matplotlib后端 ---
plt.switch_backend('Agg')

# --- 模型类定义 ---
class CoarseClassifierForLoading(nn.Module):
    """第一阶段粗分类器"""
    def __init__(self, phenossp_model, num_coarse_classes, embedding_dim=384):
        super(CoarseClassifierForLoading, self).__init__()
        self.phenossp_model = phenossp_model
        self.coarse_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_coarse_classes)
        )
    
    def forward(self, x, marker_ids=None):
        with torch.set_grad_enabled(False):
            features_dict = self.phenossp_model.forward_features(x, marker_ids=marker_ids)
            features = features_dict['x_norm_clstoken']
        coarse_logits = self.coarse_classifier(features)
        return coarse_logits

class ImmuneFineTunerForLoading(nn.Module):
    """第二阶段免疫细胞细分类器"""
    def __init__(self, phenossp_model, num_immune_fine_classes, embedding_dim=384):
        super(ImmuneFineTunerForLoading, self).__init__()
        self.phenossp_model = phenossp_model
        self.immune_fine_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_immune_fine_classes)
        )
    
    def forward(self, x, marker_ids=None):
        with torch.set_grad_enabled(False):
            features_dict = self.phenossp_model.forward_features(x, marker_ids=marker_ids)
            features = features_dict['x_norm_clstoken']
        immune_fine_logits = self.immune_fine_classifier(features)
        return immune_fine_logits

class TwoStageDataset(Dataset):
    """两阶段分类数据集"""
    def __init__(self, df, patch_dir):
        self.df = df
        self.patch_dir = patch_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_name']
        cell_id = row['cell_id']
        patch_file = f"{self.patch_dir}/{image_name}_{cell_id:06d}.h5"
        
        patch = np.zeros((7, 64, 64), dtype=np.float32)
        if os.path.exists(patch_file):
            try:
                with h5py.File(patch_file, 'r') as f:
                    channels = ['CD3', 'CD4', 'CD8', 'DAPI', 'FoxP3', 'PD1', 'PanCK']
                    for i, channel in enumerate(channels):
                        if channel in f:
                            channel_data = f[channel][:].astype(np.float32)
                            if channel_data.max() > 0:
                                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                            patch[i] = channel_data
            except Exception as e:
                print(f"读取patch文件失败 {patch_file}: {e}")
        
        patch = torch.FloatTensor(patch.copy())
        row_dict = row.to_dict()
        return patch, row_dict

def custom_collate_fn(batch):
    """自定义collate函数"""
    patches = torch.stack([item[0] for item in batch])
    row_data = [item[1] for item in batch]
    return patches, row_data

# --- 两阶段分类器 ---
class TwoStageClassifier:
    """两阶段分类器"""
    def __init__(self, coarse_model, immune_model, coarse_label_to_idx, immune_fine_label_to_idx, device):
        self.coarse_model = coarse_model
        self.immune_model = immune_model
        self.coarse_label_to_idx = coarse_label_to_idx
        self.immune_fine_label_to_idx = immune_fine_label_to_idx
        self.device = device
        
        # 创建反向映射
        self.idx_to_coarse_label = {v: k for k, v in coarse_label_to_idx.items()}
        self.idx_to_immune_fine_label = {v: k for k, v in immune_fine_label_to_idx.items()}
        
        # 创建最终细胞类型映射
        self.final_cell_type_mapping = {
            'Epithelial': 'Epithelial cell',
            'Other': 'other'
        }
        
        # 为免疫细胞类型添加映射（注意：这里使用的是过滤后的免疫细胞类型）
        immune_fine_labels = ['CD3+CD4+CD8-', 'CD3+CD4-CD8-', 'CD3-CD4+CD8-', 'CD3+CD4-CD8+', 'CD4+ Treg', 'PANCK+CD3+']
        for immune_type in immune_fine_labels:
            if immune_type == 'CD3+CD4+CD8-':
                self.final_cell_type_mapping[immune_type] = 'CD3+CD4+CD8- cell'
            elif immune_type == 'CD3+CD4-CD8-':
                self.final_cell_type_mapping[immune_type] = 'CD3+CD4-CD8- cell'
            elif immune_type == 'CD3-CD4+CD8-':
                self.final_cell_type_mapping[immune_type] = 'CD3-CD4+CD8- cell'
            elif immune_type == 'CD3+CD4-CD8+':
                self.final_cell_type_mapping[immune_type] = 'CD3+CD4-CD8+ cell'
            elif immune_type == 'CD4+ Treg':
                self.final_cell_type_mapping[immune_type] = 'CD4+ Treg'
            elif immune_type == 'PANCK+CD3+':
                self.final_cell_type_mapping[immune_type] = 'PANCK+CD3+ cell'
    
    def predict_single(self, patch):
        """预测单个样本"""
        self.coarse_model.eval()
        self.immune_model.eval()
        
        with torch.no_grad():
            # 确保patch是4维张量 (B, C, H, W)
            if patch.dim() == 3:
                patch = patch.unsqueeze(0)  # 添加batch维度
            patch = patch.to(self.device)
            
            # 创建marker_ids，对应7个通道，扩展到batch大小
            marker_ids = torch.arange(7, dtype=torch.long).unsqueeze(0).expand(patch.size(0), -1).to(self.device)
            
            # 第一阶段：粗分类
            coarse_logits = self.coarse_model(patch, marker_ids)
            coarse_probs = torch.softmax(coarse_logits, dim=1)
            coarse_pred = torch.argmax(coarse_logits, dim=1)
            coarse_label = self.idx_to_coarse_label[coarse_pred.item()]
            
            # 第二阶段：如果是免疫细胞，进行细分类
            if coarse_label == 'Immune':
                immune_logits = self.immune_model(patch, marker_ids)
                immune_probs = torch.softmax(immune_logits, dim=1)
                immune_pred = torch.argmax(immune_logits, dim=1)
                immune_fine_label = self.idx_to_immune_fine_label[immune_pred.item()]
                final_cell_type = self.final_cell_type_mapping[immune_fine_label]
                
                return {
                    'coarse_label': coarse_label,
                    'coarse_probs': coarse_probs.cpu().numpy(),
                    'immune_fine_label': immune_fine_label,
                    'immune_probs': immune_probs.cpu().numpy(),
                    'final_cell_type': final_cell_type
                }
            else:
                final_cell_type = self.final_cell_type_mapping[coarse_label]
                
                return {
                    'coarse_label': coarse_label,
                    'coarse_probs': coarse_probs.cpu().numpy(),
                    'immune_fine_label': None,
                    'immune_probs': None,
                    'final_cell_type': final_cell_type
                }
    
    def predict_batch(self, patches):
        """批量预测"""
        all_predictions = []
        
        for i in range(patches.size(0)):
            patch = patches[i]  # 获取单个patch (C, H, W)
            result = self.predict_single(patch)
            all_predictions.append(result)
        
        return all_predictions

def create_coarse_labels(df):
    """创建粗分类标签"""
    print("创建粗分类标签...")
    coarse_mapping = {
        'Epithelial': ['Epithelial cell'],
        'Immune': [
            'CD3+CD4+CD8- cell', 'CD3+CD4-CD8+ cell', 'CD3+CD4-CD8- cell',
            'CD3+CD4+CD8+ cell', 'CD3-CD4+CD8+ cell', 'CD3-CD4+CD8- cell',
            'CD3-CD4-CD8+ cell', 'PANCK+CD3+ cell', 'CD4+ Treg'
        ],
        'Other': ['other']
    }
    df['coarse_label'] = 'Other'
    for coarse_type, cell_types in coarse_mapping.items():
        df.loc[df['cell_type'].isin(cell_types), 'coarse_label'] = coarse_type
    print("粗分类标签创建完成")
    return df

def load_models(project_dir, device):
    """加载两个阶段的模型"""
    print("=== 加载两阶段模型 ===")
    
    # 加载PhenoSSP模型
    phenossp_config = { 
        "project_dir": project_dir, 
        "pretrained_path": f"{project_dir}/models/kronos_vits16_model.pt",
        "checkpoint_path": f"{project_dir}/models/kronos_vits16_model.pt",
        "model_type": "vits16", 
        "token_overlap": True,
        "num_classes": 3,
        "hf_auth_token": None,
        "cache_dir": f"{project_dir}/models/"
    }
    temp_phenossp, _, _ = CellPhenotyping(phenossp_config).load_model()
    
    # 加载第一阶段模型
    print("加载第一阶段粗分类模型...")
    coarse_model_path = f"{project_dir}/results/best_coarse_model.pth"
    if not os.path.exists(coarse_model_path):
        raise FileNotFoundError(f"找不到第一阶段模型: {coarse_model_path}")
    
    checkpoint = torch.load(coarse_model_path, map_location=device)
    coarse_label_to_idx = checkpoint.get('coarse_label_to_idx', {'Epithelial': 0, 'Immune': 1, 'Other': 2})
    
    loader_coarse_model = CoarseClassifierForLoading(temp_phenossp, 3)
    loader_coarse_model.load_state_dict(checkpoint['model_state_dict'])
    coarse_model = loader_coarse_model.to(device)
    print(f"✅ 第一阶段模型加载成功 (最佳平衡准确率: {checkpoint.get('best_balanced_acc', 'N/A')})")
    
    # 加载第二阶段模型
    print("加载第二阶段免疫细胞细分类模型...")
    immune_model_path = f"{project_dir}/results/best_immune_finetune_model.pth"
    if not os.path.exists(immune_model_path):
        raise FileNotFoundError(f"找不到第二阶段模型: {immune_model_path}")
    
    immune_checkpoint = torch.load(immune_model_path, map_location=device)
    
    # 创建免疫细胞标签映射（过滤后的6类）
    immune_fine_labels = ['CD3+CD4+CD8-', 'CD3+CD4-CD8-', 'CD3-CD4+CD8-', 'CD3+CD4-CD8+', 'CD4+ Treg', 'PANCK+CD3+']
    immune_fine_label_to_idx = {label: idx for idx, label in enumerate(immune_fine_labels)}
    
    loader_immune_model = ImmuneFineTunerForLoading(temp_phenossp, len(immune_fine_labels))
    loader_immune_model.load_state_dict(immune_checkpoint['model_state_dict'])
    immune_model = loader_immune_model.to(device)
    print(f"✅ 第二阶段模型加载成功 (最佳平衡准确率: {immune_checkpoint.get('best_balanced_acc', 'N/A')})")
    
    return coarse_model, immune_model, coarse_label_to_idx, immune_fine_label_to_idx

def evaluate_on_test_set(test_df, two_stage_classifier, project_dir):
    """在测试集上评估两阶段分类器"""
    print("\n=== 在测试集上评估两阶段分类器 ===")
    
    # 创建数据集
    patch_dir = f"{project_dir}/cell_patches_multi_with_B7"
    test_dataset = TwoStageDataset(test_df, patch_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # 进行预测
    print("进行两阶段预测...")
    all_predictions = []
    all_true_labels = []
    all_coarse_predictions = []
    all_coarse_true_labels = []
    
    for i, (patches, row_data) in enumerate(test_loader):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(test_loader)}")
        
        row = row_data[0]
        predictions = two_stage_classifier.predict_batch(patches)
        
        for pred in predictions:
            all_predictions.append(pred['final_cell_type'])
            all_true_labels.append(row['cell_type'])
            all_coarse_predictions.append(pred['coarse_label'])
            all_coarse_true_labels.append(row['coarse_label'])
    
    # 计算性能指标
    print("\n=== 两阶段分类器性能结果 ===")
    
    # 整体性能
    balanced_acc = balanced_accuracy_score(all_true_labels, all_predictions)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted')
    
    print(f"整体平衡准确率: {balanced_acc:.4f}")
    print(f"整体准确率: {accuracy:.4f}")
    print(f"F1分数 (macro): {f1_macro:.4f}")
    print(f"F1分数 (weighted): {f1_weighted:.4f}")
    
    # 粗分类性能
    coarse_balanced_acc = balanced_accuracy_score(all_coarse_true_labels, all_coarse_predictions)
    print(f"\n粗分类平衡准确率: {coarse_balanced_acc:.4f}")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_true_labels, all_predictions, zero_division=0))
    
    # 混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(all_true_labels)), 
                yticklabels=sorted(set(all_true_labels)))
    plt.title('两阶段分类器混淆矩阵 (测试集)')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_save_path = f"{project_dir}/results/two_stage_confusion_matrix_filtered.png"
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存到: {cm_save_path}")
    
    return {
        'balanced_accuracy': balanced_acc,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'coarse_balanced_accuracy': coarse_balanced_acc,
        'predictions': all_predictions,
        'true_labels': all_true_labels
    }

def main():
    """主函数"""
    print("=== 两阶段分类器推理和验证 ===")
    project_dir = "/export/home/zhangyujia/PhenoSSP_project/PhenoSSP-main/tutorials/kidney_dataset"
    
    # 加载数据
    print("加载测试数据...")
    annotation_file = f"{project_dir}/dataset/cell_annotations_A-4.csv"
    if not os.path.exists(annotation_file):
        print(f"❌ 找不到测试集注释文件: {annotation_file}")
        return
    
    test_df = pd.read_csv(annotation_file)
    test_df = create_coarse_labels(test_df)
    
    print(f"测试集样本数: {len(test_df)}")
    print("测试集细胞类型分布:")
    print(test_df['cell_type'].value_counts())
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        coarse_model, immune_model, coarse_label_to_idx, immune_fine_label_to_idx = load_models(project_dir, device)
    except FileNotFoundError as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 创建两阶段分类器
    two_stage_classifier = TwoStageClassifier(
        coarse_model, immune_model, coarse_label_to_idx, immune_fine_label_to_idx, device
    )
    
    # 在测试集上评估
    results = evaluate_on_test_set(test_df, two_stage_classifier, project_dir)
    
    print("\n🎉 两阶段分类器评估完成!")
    print(f"最终整体平衡准确率: {results['balanced_accuracy']:.4f}")
    print(f"粗分类平衡准确率: {results['coarse_balanced_accuracy']:.4f}")

if __name__ == '__main__':
    main()
