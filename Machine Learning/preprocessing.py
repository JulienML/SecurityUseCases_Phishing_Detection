import pandas as pd
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from collections import Counter, OrderedDict

nltk.download('stopwords')
nltk.download('wordnet')

def build_sender_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from the 'sender' column of the input DataFrame.
    Whole pipeline:
    - Classify sender strings into categories based on patterns (e.g., email only, name with angle brackets, etc.).
    - Create binary flags for each category (e.g., is_mail_only, is_name_angle, etc.).
    - Extract enriched features such as display name, email local part, domain, TLD, SLD, lengths, entropy, and various boolean indicators.
    - Return the DataFrame with new features added.
    """
    # Regex patterns
    RE_EMAIL_ONLY = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', re.I)
    RE_PAREN_NAME = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+\s*\([^)]+\)\s*$', re.I)
    RE_PAREN_EMPTY = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+\s*\(\s*\)\s*$', re.I)
    RE_DISPLAY_ANGLE = re.compile(r'^(?P<disp>.*?)(?P<addr><\s*[^>]+@[^>]+\s*>)\s*$', re.I)
    RE_QUOTED_NAME_ANG = re.compile(r'^"\s*[^"]+\s*"\s*<[^>]+>$', re.I)
    RE_LAST_COMMA_FIRST = re.compile(r'^"?[^",<>]+,[^",<>]+"\s*<[^>]+>$', re.I)
    NAME_CHARS = r"A-Za-zÀ-ÖØ-öø-ÿ'`’\-\. "
    RE_NAME_ANG = re.compile(rf'^[{NAME_CHARS}]+\s+[{NAME_CHARS}]+\s*<[^>]+>$')
    RE_USERNAME_STYLE = re.compile(r'^[A-Za-z0-9._-]+$')
    RE_BRACKETS = re.compile(r'\[[^\]]*\]')
    RE_FAKE_DOMAIN = re.compile(r'(no\.hostname\.specified|localhost|example\.com)', re.I)

    def _clean(s: str) -> str:
        """
        Clean input string by removing extra whitespace and line breaks.
        """
        # handle NaN/None and ensure string
        if pd.isna(s):
            return ""
        s = str(s)
        return re.sub(r'\s+', ' ', s).strip()

    def _split_display_angle(s: str) -> tuple[str, str]:
        """
        Return (display, email) when sender contains an angle-bracket address,
        otherwise return ("", ""). The returned email is without surrounding <>.
        """
        if not s:
            return "", ""
        m = RE_DISPLAY_ANGLE.search(s)
        if not m:
            return "", ""
        disp = m.group('disp').strip().strip('" ')
        addr = m.group('addr')
        # extract inner email from angle part like <a@b>
        m2 = re.search(r'<\s*([^>]+)\s*>', addr)
        email = m2.group(1).strip() if m2 else addr.strip()
        return disp, email

    cats, flag_dicts = [], []
    for sender in df["sender"]:
        
        # Creation of categorical features
        flags = OrderedDict.fromkeys([
            'is_mail_only','is_mail_paren_name','is_name_angle','is_quoted_name',
            'is_last_comma_first','is_username_angle','is_display_angle','is_multi_mails',
            'is_mail_empty_parens','is_double_at','is_mail_with_brackets','is_fake_domain','is_other',
            'is_display_empty_angle','is_quoted_text_no_email'
        ], 0)
        
        # Cleaning sender string
        sender_clean = _clean(sender)
        if not sender_clean:
            flags['is_other'] = 1
            cats.append("other")
            flag_dicts.append(flags)
            continue
        
        # Classification logic
        if "," in sender_clean and sender_clean.count("@") >= 2:
            flags['is_multi_mails'] = 1
            cats.append("multi_mails")
        elif RE_PAREN_EMPTY.match(sender_clean):
            flags['is_mail_empty_parens'] = 1
            cats.append("mail_empty_parentheses")
        elif RE_PAREN_NAME.match(sender_clean):
            flags['is_mail_paren_name'] = 1
            cats.append("mail_parentheses_lastname")
        elif RE_BRACKETS.search(sender_clean):
            flags['is_mail_with_brackets'] = 1
            cats.append("mail_with_brackets")
        elif RE_EMAIL_ONLY.match(sender_clean):
            flags['is_mail_only'] = 1
            cats.append("mail_only")
        elif sender_clean.count("@") > 1:
            flags['is_double_at'] = 1
            cats.append("mail_double_at")
        elif RE_FAKE_DOMAIN.search(sender_clean):
            flags['is_fake_domain'] = 1
            cats.append("mail_fake_domain")
        else:
            disp, angle = _split_display_angle(sender_clean)
            if angle:
                disp_core = disp.strip('" ')
                if RE_QUOTED_NAME_ANG.match(sender_clean):
                    flags['is_quoted_name'] = 1
                    cats.append("quoted_lastname_firstname_angle")
                elif RE_LAST_COMMA_FIRST.match(sender_clean):
                    flags['is_last_comma_first'] = 1
                    cats.append("lastname_comma_firstname_angle")
                elif RE_NAME_ANG.match(sender_clean):
                    flags['is_name_angle'] = 1
                    cats.append("lastname_firstname_angle")
                elif RE_USERNAME_STYLE.match(disp_core) and ' ' not in disp_core:
                    flags['is_username_angle'] = 1
                    cats.append("username_angle")
                else:
                    flags['is_display_angle'] = 1
                    cats.append("display_angle")
            
            # Special cases
            elif sender_clean.strip() in ['"" <>', '""<>']:
                flags['is_display_empty_angle'] = 1
                cats.append("display_empty_angle")
            elif sender_clean.strip().startswith('"') and sender_clean.strip().endswith('"') and '@' not in sender_clean:
                flags['is_quoted_text_no_email'] = 1
                cats.append("quoted_text_only")
            else:
                flags['is_other'] = 1
                cats.append("other")
        flag_dicts.append(flags)

    # Convert flag dicts to DataFrame and concatenate
    fmt_df = pd.DataFrame(flag_dicts)
    df["sender_category"] = cats
    df = pd.concat([df, fmt_df], axis=1)

    # Creation of enriched features
    def _shannon_entropy(s):
        if not s: return 0.0
        c = Counter(s)
        n = len(s)
        return -sum((cnt/n)*math.log2(cnt/n) for cnt in c.values())

    rows = []
    for s in df["sender"]:
        s_clean = _clean(s)
        _, email = _split_display_angle(s_clean)
        # if no angle-address, check if the cleaned string is a raw email
        if not email and RE_EMAIL_ONLY.match(s_clean):
            _, email = "", s_clean
        local, domain = ("","")
        if "@" in email:
            local, domain = email.split("@",1)
        rows.append({
            "email_local_len": len(local),
            "email_domain_len": len(domain),
            "email_local_entropy": _shannon_entropy(local),
            "email_local_has_digits": int(bool(re.search(r'\d', local))),
            "email_local_has_underscore": int("_" in local),
            "email_local_has_dot": int("." in local),
            "email_local_has_plus": int("+" in local),
            "email_domain_is_free": int(domain.lower() in {
                "gmail.com","yahoo.com","hotmail.com","outlook.com","aol.com","icloud.com",
                "protonmail.com","wanadoo.fr","orange.fr","laposte.net","free.fr","sfr.fr",
                "yandex.ru","mail.ru","zoho.com"
            }),
            "email_domain_has_digit": int(any(ch.isdigit() for ch in domain)),
            "email_domain_has_dash": int("-" in domain)
        })

    enrich_df = pd.DataFrame(rows)
    df = pd.concat([df, enrich_df], axis=1)

    return df

def preprocess_mail_content(text: str) -> str:
    """
    Preprocess email content by performing the following steps:
    1. Delete HTML tags.
    2. Replace URLs and email addresses with placeholders.
    3. Convert text to lowercase.
    4. Remove punctuation.
    5. Tokenize the text into words.
    6. Remove stop words.
    7. Lemmatize the words.
    """
    # Handle potential NaN values by converting to empty string
    if pd.isna(text):
        text = ""
    
    # 1. Delete HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Replace URLs/mails
    text = re.sub(r'http\S+|www\S+|https\S+', '<URL>', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '<EMAIL>', text)
    
    # 3. Lowercase
    text = text.lower()

    # 4. Delete punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 5. Tokenization
    tokens = text.split()

    # 6. Delete stop words
    stop_words = set(stopwords.words())
    tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)