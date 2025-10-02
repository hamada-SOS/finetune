import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

# Logging parameters
RUN_NAME = "GPT_XTTS_v2.0_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Save everything to Google Drive
OUT_PATH = "/content/drive/MyDrive/XTTS_FT_runs"

# Training Parameters (Colab safe)
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True
START_WITH_EVAL = False
BATCH_SIZE = 1              # safer for Colab T4
GRAD_ACUMM_STEPS = 252      # keep effective batch size ≈252
SAVE_STEP = 23616            # frequent saves
SAVE_N_CHECKPOINTS = 1      # keep last 3 checkpoints

# Dataset
config_dataset = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/content/coqui",
    meta_file_train="/content/coqui/metadata.csv",
    # meta_file_eval="/content/VCTK_like_dataset/metadata_eval.csv",
    language='en',
    phonemizer="so-so",
)

DATASETS_CONFIG_LIST = [config_dataset]

# Define checkpoint download directory
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# XTTS model files
TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files([TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# Speaker reference for test sentences
SPEAKER_REFERENCE = [
    "/content/coqui/wavs/sample_000001.wav"
]
LANGUAGE = config_dataset.language
def main():
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s reference
        min_conditioning_length=66150,   # 3s min
        debug_loading_failures=False,
        max_wav_length = 330750,   # ~15s
        max_text_length=267,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=2,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=23616,
        save_step=SAVE_STEP,
        save_n_checkpoints=SAVE_N_CHECKPOINTS,
        save_checkpoints=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 1e-4},

# 🔹 Learning rate schedule for 100 epochs
        lr = 1e-4,   # start high enough to adapt
        lr_scheduler = "CosineAnnealingLR",
        lr_scheduler_params = {
            "T_max": 100,     # match total epochs
            "eta_min": 5e-6   # minimum LR at the end
        },
        test_sentences=[
            {"text": "haye aboow xaalada kawarn ku dhashtay khaabuur",
             "speaker_wav": SPEAKER_REFERENCE,
             "language": LANGUAGE},
            {"text": "magacyga waa maxamed cabdi nuur adan ibrahim mursal",
             "speaker_wav": SPEAKER_REFERENCE,
             "language": LANGUAGE},
        ],
    )

    # Init model
    model = GPTTrainer.init_from_config(config)

    # Load data
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Trainer
    restore = '/content/drive/MyDrive/XTTS_FT_runs/GPT_XTTS_v2.0_LJSpeech_FT-October-02-2025_09+26AM-4f5d22b'
    
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            continue_path=restore,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

if __name__ == "__main__":
    main()


