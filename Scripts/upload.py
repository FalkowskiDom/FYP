from huggingface_hub import HfApi, upload_folder
api = HfApi()

# Upload Qwen adapter
upload_folder(
    folder_path="Outputs/qwen_logdom_3b",
    repo_id="Biplah/qwen_logdom_3b",
    repo_type="model"
)

# Upload LogBERT
# api.upload_folder(
#     folder_path="Models/logbert",
#     repo_id="Biplah/logbert_dom", 
#     repo_type="model"
# )