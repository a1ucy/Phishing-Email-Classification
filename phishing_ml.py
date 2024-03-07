# 钓鱼/诈骗邮件检测器：构建一个钓鱼/诈骗邮件检测器，利用机器学习技术来区分正常邮件和潜在的欺诈性邮件。使用公开的钓鱼邮件数据集，如Phishing Email Public Corpus (PEPC)
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


email_path = 'Phishing_Email.csv'
email_df = pd.read_csv(email_path)
data_df = email_df.dropna()

# split two texts
df_safe = email_df[email_df['Email Type']=='Safe Email']['Email Text'].astype('U')
df_phishing = email_df[email_df['Email Type']=='Phishing Email']['Email Text'].astype('U')
safe_text = df_safe.values.tolist()
phishing_text = df_phishing.values.tolist()

# balance data
min_len_email = min(len(safe_text), len(phishing_text))
safe_text = safe_text[:min_len_email]
phishing_text = phishing_text[:min_len_email]

# create label and combine data
label = [0] * min_len_email + [1] * min_len_email
emails = safe_text + phishing_text

# filter text
stop_Words = stopwords.words('english')

filtered_emails = []
for i in emails:
    email = ""
    words = word_tokenize(i)
    for word in words:
        alnum_filtered = ''.join(filter(str.isalnum, word))
        if len(alnum_filtered) > 1:
            email += alnum_filtered + ' '
    filtered_emails.append(email)

# transform and split train & test
cv = CountVectorizer(stop_words=stop_Words, binary=True)
X = cv.fit_transform(filtered_emails).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, shuffle=True)

print('start training')
# naive baye
clf = MultinomialNB()
clf.fit(X_train, y_train)
print('done training')
y_pred = clf.predict(X_test)

# calculate accuracy
score = accuracy_score(y_test, y_pred)
score_f1 = f1_score(y_test, y_pred)
y_probs = clf.predict_proba(X_test)[:, 1]
score_auc = roc_auc_score(y_test, y_probs)

print('------------------------------\nAccuracy: ', round(score,3), '\nF1: ', round(score_f1,3), '\nAUC: ', round(score_auc,3), '\n------------------------------')

# unpatch_sklearn()
