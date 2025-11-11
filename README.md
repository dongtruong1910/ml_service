ml-service/
├── data/                      # dữ liệu thô, processed, splits
│   ├── raw/                   # raw crawled posts (JSON, images)
│   ├── ocr/                   # text từ OCR (nếu dùng)
│   ├── processed/             # sau tiền xử lý (clean text, resized images)
│   └── splits/                # train/val/test csv hoặc json
│
├── ingestion/                 # scripts crawl / ingest / ETL
│   ├── crawler_fb.py
│   ├── ingest_to_s3.py
│   └── README.md
│
├── preprocessing/             # text/image preprocess scripts
│   ├── text_cleaning.py
│   ├── ocr_utils.py
│   ├── image_transforms.py
│   └── create_splits.py
│
├── datasets/                  # dataset class / dataloader
│   └── fb_dataset.py
│
├── models/                    # model definitions + checkpoints
│   ├── text/                  # PhoBERT / BERT wrappers
│   │   └── phobert_wrapper.py
│   ├── image/                 # ViT wrapper
│   │   └── vit_wrapper.py
│   ├── fusion/                # fusion + classifier head
│   │   └── fusion_model.py
│   └── loaders.py             # load pretrained weights / restore checkpoints
│
├── training/                  # training loop, config, utils
│   ├── train.py
│   ├── config.yaml
│   ├── scheduler.py
│   └── metrics.py
│
├── inference/                 # inference API logic (load model + predict)
│   └── predictor.py
│
├── api/                       # FastAPI app serving inference endpoints
│   ├── main.py
│   ├── routers/
│   │   ├── classify.py
│   │   └── health.py
│   └── schemas.py
│
├── experiments/               # notebooks, experiment logs
│
├── ops/                       # Dockerfile, docker-compose, k8s manifests
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── mlflow/                    # optional: mlflow config for experiment tracking
│
├── requirements.txt
└── README.md
