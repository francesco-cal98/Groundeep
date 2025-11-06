# Groundeep 🧠  
**Emergent Structure and Grounding in Deep Belief Networks**

> *Groundeep* is an ongoing research framework dedicated to studying how structured, interpretable representations emerge in **unsupervised Deep Belief Networks (DBN)** and how these internal representations can be **anchored (grounded)** across modalities such as vision, text, and number concepts.

---

## 🌍 Project Overview

Groundeep originates from a long-term investigation into **multimodal Deep Belief Networks** (iMDBN).  
It extends those experiments by providing a **systematic environment** for analyzing and stabilizing the dynamics of DBN training — from *unimodal pretraining* to *cross-modal grounding*.

The central research question is:

> How can low-level, modality-specific representations (e.g., images, symbols, numerosity) develop a shared latent geometry that reflects **semantic structure** — even without explicit supervision?

---

## 🚧 Current Focus

This repository captures the **intermediate state** of that exploration.  
Models, scripts, and diagnostics evolve rapidly as new hypotheses are tested.

**Current experimental directions include:**

- **Gradient and variance balancing** between latent subspaces (`z` and `y`)  
- **Warm-up and auxiliary clamping** to consolidate `p(z|y)` before free-phase contrastive divergence  
- **“Hot” Gibbs sampling** for stabilizing cross-modal generation (TXT→IMG, IMG→TXT)  
- **Dynamic gain control** aligning latent statistics (mean/variance) across modalities  
- **Representation diagnostics** using cosine similarity, variance analysis, entropy tracking, and RSA  
- **Mode-collapse mitigation** and cross-modal equilibrium tuning  

---

## 🧩 Conceptual Architecture
    [ Image Encoder DBN ]          [ Text Encoder RBM ]
               ↓                           ↓
       z_img latent space          z_txt latent space
               ↓                           ↓
        └─────── Joint RBM (z_joint) ───────┘
                    ↑           ↑
             Reconstruction and Gibbs flows

    [ Image Encoder DBN ]          [ Text Encoder RBM ]
               ↓                           ↓
       z_img latent space          z_txt latent space
               ↓                           ↓
        └─────── Joint RBM (z_joint) ───────┘
                    ↑           ↑
             Reconstruction and Gibbs flows



