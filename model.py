from transformers import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaConfig,
    AutoModel,    
)
import torch
import torch.nn as nn
import os


class LlavaQformerForConditionalGeneration(LlavaForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaConfig):
        super(LlavaForConditionalGeneration, self).__init__(
            config
        )  # 跳过父类的__init__
        self.model = LlavaQformerModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        # 确保权重绑定
        self.tie_weights()
        self.post_init()

    def tie_weights(self):
        """绑定lm_head和embed_tokens的权重"""
        if hasattr(self.model.language_model, "embed_tokens"):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight


class LlavaQformerModel(LlavaModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaConfig):
        super(LlavaModel, self).__init__(config)  # 跳过父类的__init__
        self.vision_tower = AutoModel.from_config(config.vision_config)

        if hasattr(config, "qformer_config"):
            qformer_config = config.qformer_config
            qformer = QueryTransformer(**qformer_config)
        else:
            qformer = QueryTransformer()

        self.multi_modal_projector = qformer
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()


# 定义QueryTransformer类
class QueryTransformer(nn.Module):
    def __init__(
        self,
        use_qformer=True,
        num_query_tokens=32,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        input_dim=1024,  # 来自vision_config.hidden_size
        output_dim=4096,  # 来自text_config.hidden_size
    ):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range  # 保存初始化范围
        
        # 可学习的query embeddings
        self.query_embeddings = nn.Parameter(
            torch.randn(num_query_tokens, hidden_size) * initializer_range
        )
        
        # 图像特征投影到Q-Former hidden size
        self.image_projection = nn.Linear(input_dim, hidden_size)
        
        # Q-Former layers
        self.qformer_layers = nn.ModuleList([
            QFormerLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                layer_norm_eps=layer_norm_eps
            ) for _ in range(num_hidden_layers)
        ])
        
        # 输出投影到语言模型的hidden size
        self.output_projection = nn.Linear(hidden_size, output_dim)
        self.output_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _initialize_weights(self, module):
        """为transformers兼容性添加的权重初始化方法"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.initializer_range)

    def _init_weights(self, initializer_range):
        """原有的初始化方法"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, image_features):
        # image_features shape: [batch_size, seq_len, input_dim]
        batch_size = image_features.size(0)
        
        # 投影图像特征
        image_features = self.image_projection(image_features)
        
        # 扩展query embeddings到batch
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 通过Q-Former层
        for layer in self.qformer_layers:
            queries = layer(queries, image_features)
        
        queries = self.output_norm(queries)
        queries = self.output_projection(queries)
        
        return queries  # shape: [batch_size, num_query_tokens, output_dim]

class QFormerLayer(nn.Module):
    def __init__(
        self, 
        hidden_size, 
        num_attention_heads, 
        intermediate_size,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        layer_norm_eps
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self-attention for queries
        self.self_attention = nn.MultiheadAttention(
            hidden_size, 
            num_attention_heads, 
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )
        # Cross-attention: queries attend to image features
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, 
            num_attention_heads, 
            dropout=attention_probs_dropout_prob,
            batch_first=True
        )
        # Feed forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(hidden_dropout_prob)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _initialize_weights(self, module):
        """为transformers兼容性添加的权重初始化方法"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, queries, image_features):
        # Self-attention
        attn_output, _ = self.self_attention(queries, queries, queries)
        queries = self.norm1(queries + attn_output)
        
        # Cross-attention
        attn_output, _ = self.cross_attention(queries, image_features, image_features)
        queries = self.norm2(queries + attn_output)
        
        # FFN
        ffn_output = self.ffn(queries)
        queries = self.norm3(queries + ffn_output)
        
        return queries
