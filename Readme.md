This project is a proof-of-concept that demonstrates how AI can be used in Log analysis.

To run this code you will need a both a logBERT model and an LLM
Already trained logBERT model available at https://huggingface.co/Biplah/logbert_dom 

Trained LLM available at https://huggingface.co/Biplah/qwen_logdom_3b_merged This code can also work on a standard LLM with no training

Python 3.12 or under is need to run this code as any python version after 3.12 does not have a GPU Pytorch version 
Python 3.11.9 recommended

To run the code you need 3 terminals:
1. Run the API server: `python -m uvicorn Src.App.app:app --reload`
2. Run the real-time ingestion: `python Src\App\realtime_ingest.py`
3. Run the GUI: `streamlit run Src\App\gui.py`
