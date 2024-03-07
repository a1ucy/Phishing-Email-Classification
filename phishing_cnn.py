import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import re

warnings.filterwarnings("ignore")

# constant
min_freq = 5
len_text = 200
batch_size = 64
num_epochs = 5
dropout = 0.2
embedding_dim = 64
hidden_size = 100
kernel_sizes = [3, 4, 5]
num_filters = 100

stop_Words = stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
email_path = 'Phishing_Email.csv'

# clean data
email_df = pd.read_csv(email_path, header=None).dropna()

email_df['category'] = email_df[2].astype('category')
email_df['label'] = email_df['category'].cat.codes
df = email_df[email_df[1] != 'empty']
df = df.iloc[:, [1, -1]]

# balance data
min_len = df['label'].value_counts().min()
df_class_0 = df[df['label'] == 0].sample(min_len)
df_class_1 = df[df['label'] == 1].sample(min_len)
df = pd.concat([df_class_0, df_class_1])

# filter words, remove punctuations and stop words
def filter_content(content):
    email = []
    words = word_tokenize(content)
    for word in words:
        word = word.lower()
        alnum_filtered = re.sub(r'[^a-zA-Z0-9]', '', word)
        if len(alnum_filtered) > 1 and alnum_filtered not in stop_Words:
            email.append(alnum_filtered)
    return email

df[1] = df[1].apply(filter_content)

# df to list
contents = df[1]
labels = df['label']
contents = contents.values.tolist()
labels = labels.values.tolist()
labels = [int(item) for item in labels]

# split data
X_train, X_test, y_train, y_test = train_test_split(contents, labels, test_size=0.2, shuffle=True)

# build vocab
vocab = build_vocab_from_iterator(contents, min_freq=min_freq, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# email text to index and pad
def txt_to_index(emails):
    result = []
    for i in emails:
        idx = vocab(i)
        if len(idx) > len_text:
            idx = idx[:len_text]
        elif len(idx) < len_text:
            idx += [0] * (len_text - len(idx))
        result.append(idx)
    return result


X_train_idx = txt_to_index(X_train)
X_test_idx = txt_to_index(X_test)

# transform type to tensor
X_train_idx = torch.tensor(X_train_idx)
X_test_idx = torch.tensor(X_test_idx)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# pack data to loader
train_dataset = TensorDataset(X_train_idx, y_train)
test_dataset = TensorDataset(X_test_idx, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(len(kernel_sizes) * num_filters, 1)

    def forward(self, text):
        x = self.embedding(text)  # [batch_size, text_len, emb_dim]
        x = x.permute(0, 2, 1)  # [bs, ed, sl]
        conved = [F.relu(conv(x)) for conv in self.convs]  # conved[0] = [bs, num_filter, sl-2]
        # 200 (embedding_dim) - 3 (kernel_size) + 1 = 198.
        # [bs, num_filter, 198] -> max pool -> [bs, num_filter, 1] drop 1
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # pooled[0] = [bs, num_filter]
        output = self.dropout(torch.cat(pooled, dim=1))  # [bs, num_filter * len(kernels)]
        output = self.linear(output)  # [bs,1]
        return torch.sigmoid(output).squeeze()


num_embedding = len(vocab)
model = CNN().to(device)

# loss & optimizer
loss_fn = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    num = 0
    for texts, labels in train_loader:
        # Forward pass
        texts = texts.to(device)
        labels = labels.to(device)
        outputs = model(texts)
        outputs = outputs.float()
        labels = labels.float()
        loss = loss_fn(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test the model
    model.eval()
    total_val_loss = 0
    all_labels = []
    all_predictions = []
    all_outputs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            # print(type(texts))
            outputs = model(texts)
            outputs = outputs.float()
            labels = labels.float()

            loss = loss_fn(outputs, labels)
            total_val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.squeeze().cpu().numpy())

    avg_val_loss = total_val_loss / len(test_loader)
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc = roc_auc_score(all_labels, all_outputs)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Accuracy: {(correct / total):.4f}, F1 score: {f1:.4f}, AUC: {auc:.4f}')

    torch.save(model.state_dict(), 'cnn_model.pt')

# # test model with own email
# model.load_state_dict(torch.load('./cnn_model.pt'))
# def single_email_test(text):
#     test_text = filter_content(text)
#     test_text = [test_text]
#     for i in range(batch_size-1):
#         test_text.append(['a'])
#     test_text = txt_to_index(test_text)
#     test_text = torch.tensor(test_text)
#     y = model(test_text.to(device))
#     result = (y[0].squeeze() > 0.5).float()
#     print('这是钓鱼邮件。' if result == torch.tensor(0).float() else '这是正常邮件。')
#
# spam_text = "Use each coupon up to 5 times in 1 transaction.1 Clip CouponsYour Weekly Featured Coupons: Kroger Brand Ultra Paper Towels, General Mills Honey Nut Cheerios, Lay's Classic Potato Chips Don’t Forget: SAVE an Extra $10 When you buy 2 participating Household items, with your Card.3 Shop Now These deals aren’t going anywhere. Find locked-in savings on favorites in-store and online. Look for the lock symbol in-store. Enjoy FREE grocery delivery†, 2X Fuel Points^ and more. Start Your 30-day Free Trial‡ » Already a Boost member? Enjoying your Free Trial? Review your membership details"
#
# safe_text = "The password for your DMV account has changed. Please use your new password to log in. Get your renewal notices electronically. Link your driver’s license/identification card to your account and visit https://www.dmv.ca.gov/portal/paperless-notices/ If you have any questions, please call 1-877-563-5213. This email was sent from an unattended mailbox. Please do not respond using the Reply button."
#
# single_email_test(spam_text)
# single_email_test(safe_text)

