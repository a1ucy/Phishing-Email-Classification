import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# constant
stop_Words = stopwords.words('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 0.001
num_epochs = 3
max_features = 1000

email_path = 'Phishing_Email.csv'
email_df = pd.read_csv(email_path)
data_df = email_df.dropna()

# split two texts
df_safe = email_df[email_df['Email Type'] == 'Safe Email']['Email Text'].astype('U')
df_phishing = email_df[email_df['Email Type'] == 'Phishing Email']['Email Text'].astype('U')
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
def filter_content(content):
    email = ""
    words = word_tokenize(content)
    for word in words:
        alnum_filtered = ''.join(filter(str.isalnum, word))
        if len(alnum_filtered) > 1 and word not in stop_Words:
            email += alnum_filtered + ' '
    return email


filtered_emails = []
for i in emails:
    email = filter_content(i)
    filtered_emails.append(email)

# transform and split train & test
cv = TfidfVectorizer(stop_words=stop_Words, binary=True, max_features=max_features)
X = cv.fit_transform(filtered_emails).toarray()
X = torch.tensor(X, dtype=torch.float32)
labels = torch.tensor(label, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True)


class EmailDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = EmailDataset(X_train, y_train)
test_dataset = EmailDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Model(nn.Module):
    def __init__(self, max_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(max_features, 32)
        self.output = nn.Linear(32, 1)
        self.af = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(self.af(self.fc1(x)))
        # x = self.dropout(self.af(self.fc2(x)))
        # x = self.dropout(self.af(self.fc3(x)))
        # x = self.dropout(self.af(self.fc4(x)))
        # x = self.dropout(self.af(self.fc5(x)))
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


model = Model(max_features).to(device)

# Loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for texts, labels in train_loader:
        # Forward pass
        outputs = model(texts)
        outputs = outputs.reshape(-1)
        loss = loss_fn(outputs.float(), labels.float())

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
            outputs = model(texts)
            outputs = outputs.reshape(-1)
            loss = loss_fn(outputs.float(), labels.float())
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
    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss:.4f}, Accuracy: {(100 * correct / total):.2f}%, F1 score: {f1:.4f}, AUC: {auc:.4f}')
