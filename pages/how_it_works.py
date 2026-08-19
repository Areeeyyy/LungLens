import streamlit as st

def page_how_it_works():
    st.markdown(
        """
        <div class="upload-header">
            <h2>How It Works?</h2>
            <p>A look under the hood of LungLens's pneumonia detection pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Overview ---
    st.markdown(
        """
        <div class="hiw-section">
            <h2>Overview</h2>
            <p>
                LungLens uses a <strong>Convolutional Neural Network (CNN)</strong>, a type
                of deep learning architecture specifically designed for processing spatial
                image data, to classify chest X-ray images as either
                <strong>NORMAL</strong> or <strong>PNEUMONIA</strong>.
            </p>
            <p>
                The model was trained on the
                <em>Chest X-Ray Images (Pneumonia)</em> dataset from Kaggle, following the
                <strong>CRISP-DM</strong> (Cross-Industry Standard Process for Data Mining)
                methodology.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- CNN Architecture ---
    st.markdown(
        """
        <div class="hiw-section">
            <h2>CNN Architecture</h2>
            <p>
                The model uses a <strong>Sequential CNN</strong> with five convolutional
                blocks followed by a classifier head. Each block performs feature extraction
                by applying convolution filters, activation, and pooling, with filters
                increasing from 32 to 256 to capture progressively complex patterns.
            </p>
            <div class="arch-block">
                Input (150 × 150 × 3, RGB Image)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Conv2D(32, 3×3) &nbsp;→ ReLU → MaxPool(2×2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Conv2D(64, 3×3) &nbsp;→ ReLU → MaxPool(2×2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Conv2D(128, 3×3) → ReLU → MaxPool(2×2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Conv2D(128, 3×3) → ReLU → MaxPool(2×2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Conv2D(256, 3×3) → ReLU → MaxPool(2×2)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Flatten<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─ Dense(512, ReLU) → Dropout(0.5)<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;└─ Dense(1, Sigmoid) → Output [0…1]<br>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Layer explanations ---
    st.markdown(
        '<div class="hiw-section"><h2>Key Layers Explained</h2></div>',
        unsafe_allow_html=True,
    )

    layers = [
        (
            "Convolutional Layer (Conv2D)",
            "The core layer that performs mathematical convolution to extract features. "
            "Convolution multiplies pixel value matrices element-wise with a learnable "
            "filter (kernel), sliding across the image to produce a feature map that "
            "highlights patterns such as edges, textures, and shapes. "
            "This model uses five Conv2D blocks with increasing filters: 32 → 64 → 128 → 128 → 256.",
        ),
        (
            "Activation Function (ReLU)",
            "The Rectified Linear Unit introduces non-linearity to the network. "
            "Mathematically defined as <strong>f(x) = max(0, x)</strong>, it sets all "
            "negative values to zero while keeping positive values unchanged, "
            "enabling the network to learn complex patterns.",
        ),
        (
            "Pooling Layer (MaxPooling2D)",
            "Reduces the spatial dimensions (width and height) of feature maps "
            "without losing the most important information. Max pooling takes "
            "the maximum value from each window region, shrinking the data while "
            "preserving dominant features.",
        ),
        (
            "Dropout (0.5)",
            "A regularization technique that randomly deactivates 50% of neurons "
            "during training, preventing the model from memorizing (overfitting) "
            "and encouraging it to learn more robust features. "
            "Applied once after the Dense(512) layer.",
        ),
        (
            "Dense Layer &amp; Sigmoid Output",
            "The fully connected Dense(512) layer with ReLU activation weighs "
            "probabilities from the flattened features. "
            "The final neuron uses a <strong>sigmoid</strong> activation to output a "
            "probability between 0 (Normal) and 1 (Pneumonia).",
        ),
    ]

    for title, desc in layers:
        st.markdown(
            f"""
            <div style="margin-bottom:1.25rem;">
                <h3 style="font-size:1.1rem; font-weight:700; color:var(--text-primary);
                    margin:0 0 0.4rem; font-family:'Inter',sans-serif;">{title}</h3>
                <p style="font-size:0.95rem; color:var(--text-secondary); line-height:1.7;
                    margin:0; font-family:'Inter',sans-serif;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- CRISP-DM Pipeline ---
    st.markdown(
        """
        <div class="hiw-section">
            <h2>Training Pipeline (CRISP-DM)</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    steps = [
        (
            "Data Understanding",
            "Analyze the chest X-ray dataset distribution, identify class imbalance "
            "between NORMAL and PNEUMONIA images.",
        ),
        (
            "Data Splitting",
            "80/20 stratified split performed <strong>before</strong> oversampling "
            "to prevent data leakage and ensure unbiased evaluation.",
        ),
        (
            "Data Preparation",
            "Handle class imbalance using <strong>class weight</strong> "
            "(<code>compute_class_weight</code>) and apply augmentation "
            "(rotation, zoom, horizontal flip, width/height shift) on all training data. "
            "Images resized to 150×150 and normalized to [0, 1].",
        ),
        (
            "Modeling",
            "Build the Sequential CNN architecture (Conv2D 32→64→128→128→256, Dense 512→1) "
            "compiled with Adam optimizer and binary cross-entropy loss.",
        ),
        (
            "Training",
            "Train with EarlyStopping (patience=5), ModelCheckpoint (save best), "
            "and ReduceLROnPlateau callbacks for optimal convergence.",
        ),
        (
            "Evaluation",
            "Assess performance on the held-out 20% test set using Accuracy, "
            "Precision, Recall, F1-Score, and Confusion Matrix.",
        ),
    ]

    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-num">{i}</div>
                <div class="step-body">
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- How Prediction Works ---
    st.markdown(
        """
        <div class="hiw-section" style="margin-top:2rem;">
            <h2>How Prediction Works?</h2>
            <p>When you upload an X-ray image, LungLens performs these steps:</p>
            <ul>
                <li><strong>Preprocessing</strong>, The image is converted to RGB,
                    resized to 150×150 pixels, and pixel values are normalized to the
                    range [0, 1].</li>
                <li><strong>Inference</strong>, The preprocessed image is fed through
                    the CNN model, which outputs a single probability value.</li>
                <li><strong>Classification</strong>, If the probability ≥ 0.5, the
                    image is classified as <strong>PNEUMONIA</strong>; otherwise, as
                    <strong>NORMAL</strong>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Why it detects pneumonia ---
    st.markdown(
        """
        <div class="hiw-section">
            <h2>Why CNN Can Detect Pneumonia</h2>
            <p>
                Although chest X-rays are a common diagnostic tool, manual interpretation
                by radiologists can be subjective and error-prone in subtle cases.
            </p>
            <p>
                CNN excels at detecting changes in <strong>grayscale gradation and opacity</strong>
                in X-ray images. Convolutional filters are highly sensitive to shifts in
                intensity that indicate <em>infiltrates</em>, the hazy white patches in
                lung regions characteristic of pneumonia. These are patterns that may be
                difficult for the human eye to consistently identify across thousands of images.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
