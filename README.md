# DataEngineering

To execution:
pip install streamlit
[window configure env.] - may need to confgure window
python -m pip install cryptography
python -m pip install langchain
python -m pip install langchain_community
pip install openai
pip install lagent

pip install requests beautifulsoup4
pip install sumy

python -c "import nltk; nltk.download('punkt')"
python -m nltk.downloader all

streamlit run testv1.X.py


Yungeng：

I make evaluation:

Falcon 11B
{
    "predict_bleu-4": 4.2507947,
    "predict_model_preparation_time": 0.0044,
    "predict_rouge-1": 13.814779,
    "predict_rouge-2": 4.3531343,
    "predict_rouge-l": 9.8814781,
    "predict_runtime": 2691.25,
    "predict_samples_per_second": 0.372,
    "predict_steps_per_second": 0.046
}

InternLM2-7B
{
    "predict_bleu-4": 5.114347599999999,
    "predict_model_preparation_time": 0.0027,
    "predict_rouge-1": 21.361561000000002,
    "predict_rouge-2": 6.683733899999999,
    "predict_rouge-l": 14.5170265,
    "predict_runtime": 1943.6994,
    "predict_samples_per_second": 0.514,
    "predict_steps_per_second": 0.064
}

Yi-9B
{
    "predict_bleu-4": 4.1292837,
    "predict_model_preparation_time": 0.0045,
    "predict_rouge-1": 18.3583215,
    "predict_rouge-2": 4.7908943,
    "predict_rouge-l": 12.552900900000001,
    "predict_runtime": 2893.4876,
    "predict_samples_per_second": 0.346,
    "predict_steps_per_second": 0.043
}

Qwen-2.5B

{
    "predict_bleu-4": 11.772052100000002,
    "predict_model_preparation_time": 0.0039,
    "predict_rouge-1": 35.3489205,
    "predict_rouge-2": 14.0041114,
    "predict_rouge-l": 27.1539784,
    "predict_runtime": 2199.6616,
    "predict_samples_per_second": 0.455,
    "predict_steps_per_second": 0.057
}
