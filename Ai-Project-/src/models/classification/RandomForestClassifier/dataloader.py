import pandas as pd

def load_data(path="DatasetCredit-g.csv"):
    df = pd.read_csv(path)

    # -------------------------------
    # 1. SELEÇÃO
    # -------------------------------
    df = df.drop(columns=['other_parties', 'foreign_worker', 'gender'])

    # -------------------------------
    # 2. CONSTRUÇÃO
    # -------------------------------
    # Merge credit history values
    df['credit_history'] = df['credit_history'].replace({
        'all paid': 'all_paid',
        'no credits/all paid': 'all_paid'
    })

    # Rare categories in purpose (<5%) -> "other"
    freq = df['purpose'].value_counts(normalize=True)
    rare_purposes = freq[freq < 0.05].index
    df['purpose'] = df['purpose'].replace(rare_purposes, 'other')

    # Remove existing_credits values 3 and 4
    df = df[~df['existing_credits'].isin([3, 4])]


    # Remove age > 65
    df = df[df['age'] <= 65]

    # -------------------------------
    # 3. FORMATAÇÃO
    # -------------------------------
    # Binary encoding
    df['existing_credits_bin'] = (df['existing_credits'] >= 2).astype(int)
    df['own_telephone_bin'] = (df['own_telephone'] == 'yes').astype(int)

    # One-hot encoding
    one_hot_cols = [
        'checking_status', 'credit_history', 'purpose', 'savings_status',
        'employment', 'personal_status', 'property_magnitude',
        'other_payment_plans', 'housing', 'job'
    ]
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Label encoding target
    df['class'] = df['class'].map({'bad': 0, 'good': 1})

    # -------------------------------
    # Final split
    # -------------------------------
    X = df.drop(columns=['class', 'existing_credits', 'own_telephone'])
    y = df['class']

    return X, y
