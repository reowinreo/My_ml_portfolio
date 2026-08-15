if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision import models, datasets, transforms
    import numpy as np
    import os
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from torch.utils.data import Subset, DataLoader

    data_dir = 'dataset_raw'
    split_path = 'saved_models/split_indices.npz'
    model_path = 'saved_models/no_label_smoothing_final_model.pth'
    batch_size = 128
    num_classes = 45
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_data = np.load(split_path)
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, data_transforms)
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    def extract_features(dataloader):
        features_list, labels_list = [], []

        def hook(module, input, output):
            features_list.append(output.detach().cpu())

        handle = model.avgpool.register_forward_hook(hook)

        all_features, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels_list.extend(labels.numpy())
                _ = model(inputs)
                batch_features = torch.cat(features_list, dim=0)
                features_list.clear()
                batch_features = batch_features.view(batch_features.size(0), -1)
                all_features.append(batch_features)
            all_features = torch.cat(all_features, dim=0)
        handle.remove()
        return all_features.cpu().numpy(), np.array(labels_list)

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use training-set mean and variance

    pca = PCA()
    pca.fit(X_train_scaled)
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(explained_variance_ratio >= 0.95) + 1
    print(f"Principal components explaining 95% variance: {k}")

    X_train_pca = pca.transform(X_train_scaled)[:, :k]
    X_test_pca = pca.transform(X_test_scaled)[:, :k]

    print("Training SVM")
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train_pca, y_train)

    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {acc:.4f}")
