import nltk
import ssl, certifi
_create_unverified_https_context = ssl._create_unverified_context
# ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")
nltk.download('vader_lexicon')