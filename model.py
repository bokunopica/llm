from transformers import (
    LlavaForConditionalGeneration,
    LlavaModel,
    LlavaConfig,
    AutoModel,
)
import torch.nn as nn


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
        input_dim=1024,
        hidden_dim=4096,
        output_dim=4096,
        num_heads=8,
        num_layers=2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.norm(x)
        x = self.transformer(x)
        x = self.output_projection(x)
        return x
