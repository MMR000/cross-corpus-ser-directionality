# Paper Implementation Details

Values are parsed from the YAML configs used for experiments, with selected code-level defaults filled in when not explicit.

Code-derived defaults used:
- optimizer = `AdamW` from `src/training/trainer.py` (`SERTrainer.__init__`)
- dropout default = `0.2` in model classes
- num_classes default = `4` in model classes
- wav2vec2 freeze default = `True` in `src/features/audio_features.py` (`build_frontend` / `Wav2Vec2Frontend`)
- GRL alpha default = `1.0` when not explicit

Total configs audited: 41