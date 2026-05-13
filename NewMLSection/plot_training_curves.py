import matplotlib.pyplot as plt
import re

log_data = """
Epoch 001 | Train Loss: 0.9845 | Val MSE: 26036004.0000 | Val R2: -0.8048
Epoch 002 | Train Loss: 0.8572 | Val MSE: 27654512.0000 | Val R2: -1.7925
Epoch 003 | Train Loss: 0.8256 | Val MSE: 29714352.0000 | Val R2: -1.4809
Epoch 004 | Train Loss: 0.8089 | Val MSE: 27096386.0000 | Val R2: -0.6363
Epoch 005 | Train Loss: 0.7943 | Val MSE: 24775592.0000 | Val R2: 0.0816
Epoch 006 | Train Loss: 0.7862 | Val MSE: 25678952.0000 | Val R2: 0.1208
Epoch 007 | Train Loss: 0.7775 | Val MSE: 25944008.0000 | Val R2: 0.0327
Epoch 008 | Train Loss: 0.7825 | Val MSE: 26509696.0000 | Val R2: -0.2771
Epoch 009 | Train Loss: 0.7732 | Val MSE: 30392450.0000 | Val R2: -1.2357
Epoch 010 | Train Loss: 0.7707 | Val MSE: 27088246.0000 | Val R2: -0.1295
Epoch 011 | Train Loss: 0.7734 | Val MSE: 30057376.0000 | Val R2: -0.5090
Epoch 012 | Train Loss: 0.7740 | Val MSE: 27855370.0000 | Val R2: -0.0819
Epoch 013 | Train Loss: 0.7701 | Val MSE: 26399450.0000 | Val R2: -0.0830
Epoch 014 | Train Loss: 0.7714 | Val MSE: 24888282.0000 | Val R2: 0.1349
Epoch 015 | Train Loss: 0.7706 | Val MSE: 26810988.0000 | Val R2: -0.0818
Epoch 016 | Train Loss: 0.7638 | Val MSE: 29616966.0000 | Val R2: -0.7730
Epoch 017 | Train Loss: 0.7620 | Val MSE: 31245966.0000 | Val R2: -1.0522
Epoch 018 | Train Loss: 0.7547 | Val MSE: 27396494.0000 | Val R2: -0.3842
Epoch 019 | Train Loss: 0.7613 | Val MSE: 30658126.0000 | Val R2: -1.3105
Epoch 020 | Train Loss: 0.7457 | Val MSE: 33151464.0000 | Val R2: -0.9070
Epoch 021 | Train Loss: 0.7489 | Val MSE: 29860948.0000 | Val R2: -0.9439
Epoch 022 | Train Loss: 0.7468 | Val MSE: 44678264.0000 | Val R2: -3.7771
Epoch 023 | Train Loss: 0.7456 | Val MSE: 27929354.0000 | Val R2: -0.2850
Epoch 024 | Train Loss: 0.7478 | Val MSE: 26829586.0000 | Val R2: -0.1079
Epoch 025 | Train Loss: 0.7394 | Val MSE: 28685458.0000 | Val R2: -0.2986
Epoch 026 | Train Loss: 0.7344 | Val MSE: 28706152.0000 | Val R2: -0.5107
Epoch 027 | Train Loss: 0.7336 | Val MSE: 32496794.0000 | Val R2: -0.7918
Epoch 028 | Train Loss: 0.7354 | Val MSE: 27100684.0000 | Val R2: -0.1685
Epoch 029 | Train Loss: 0.7404 | Val MSE: 25209516.0000 | Val R2: 0.0465
Epoch 030 | Train Loss: 0.7434 | Val MSE: 27718616.0000 | Val R2: -0.1975
Epoch 031 | Train Loss: 0.7586 | Val MSE: 26592044.0000 | Val R2: -0.1735
Epoch 032 | Train Loss: 0.7422 | Val MSE: 30709178.0000 | Val R2: -0.2022
Epoch 033 | Train Loss: 0.7365 | Val MSE: 27359114.0000 | Val R2: -0.0518
Epoch 034 | Train Loss: 0.7485 | Val MSE: 25970456.0000 | Val R2: -0.2520
Epoch 035 | Train Loss: 0.7474 | Val MSE: 28597250.0000 | Val R2: -0.3848
Epoch 036 | Train Loss: 0.7452 | Val MSE: 25370540.0000 | Val R2: 0.0700
Epoch 037 | Train Loss: 0.7386 | Val MSE: 27266422.0000 | Val R2: -0.3268
Epoch 038 | Train Loss: 0.7427 | Val MSE: 29964550.0000 | Val R2: -0.9352
Epoch 039 | Train Loss: 0.7338 | Val MSE: 27277556.0000 | Val R2: -0.1380
Epoch 040 | Train Loss: 0.7324 | Val MSE: 29160406.0000 | Val R2: -0.4193
Epoch 041 | Train Loss: 0.7333 | Val MSE: 27451340.0000 | Val R2: -0.1730
Epoch 042 | Train Loss: 0.7335 | Val MSE: 28982954.0000 | Val R2: -0.4138
Epoch 043 | Train Loss: 0.7350 | Val MSE: 26475620.0000 | Val R2: -0.1109
Epoch 044 | Train Loss: 0.7357 | Val MSE: 28901488.0000 | Val R2: -0.4954
Epoch 045 | Train Loss: 0.7386 | Val MSE: 27791882.0000 | Val R2: -0.2752
Epoch 046 | Train Loss: 0.7363 | Val MSE: 26552484.0000 | Val R2: -0.1380
Epoch 047 | Train Loss: 0.7353 | Val MSE: 33048100.0000 | Val R2: -1.2589
Epoch 048 | Train Loss: 0.7366 | Val MSE: 26155120.0000 | Val R2: -0.1856
Epoch 049 | Train Loss: 0.7343 | Val MSE: 29540694.0000 | Val R2: -0.7290
Epoch 050 | Train Loss: 0.7280 | Val MSE: 29523998.0000 | Val R2: -0.3834
Epoch 051 | Train Loss: 0.7311 | Val MSE: 26607802.0000 | Val R2: -0.2787
Epoch 052 | Train Loss: 0.7305 | Val MSE: 26602942.0000 | Val R2: -0.4349
Epoch 053 | Train Loss: 0.7319 | Val MSE: 25811666.0000 | Val R2: -0.0160
Epoch 054 | Train Loss: 0.7314 | Val MSE: 28208664.0000 | Val R2: -0.1933
Epoch 055 | Train Loss: 0.7330 | Val MSE: 26257352.0000 | Val R2: -0.2116
Epoch 056 | Train Loss: 0.7302 | Val MSE: 28703858.0000 | Val R2: -0.4993
Epoch 057 | Train Loss: 0.7271 | Val MSE: 28712590.0000 | Val R2: -0.5760
Epoch 058 | Train Loss: 0.7258 | Val MSE: 28532294.0000 | Val R2: -0.5252
Epoch 059 | Train Loss: 0.7267 | Val MSE: 27810864.0000 | Val R2: -0.2860
Epoch 060 | Train Loss: 0.7288 | Val MSE: 27853028.0000 | Val R2: -0.3627
Epoch 061 | Train Loss: 0.7267 | Val MSE: 27983860.0000 | Val R2: -0.6311
Epoch 062 | Train Loss: 0.7264 | Val MSE: 28941404.0000 | Val R2: -0.5476
Epoch 063 | Train Loss: 0.7281 | Val MSE: 28500158.0000 | Val R2: -0.7453
Epoch 064 | Train Loss: 0.7290 | Val MSE: 26386966.0000 | Val R2: -0.2592
Epoch 065 | Train Loss: 0.7288 | Val MSE: 28109228.0000 | Val R2: -0.5592
Epoch 066 | Train Loss: 0.7237 | Val MSE: 28844502.0000 | Val R2: -0.5593
Epoch 067 | Train Loss: 0.7226 | Val MSE: 28452814.0000 | Val R2: -0.4893
Epoch 068 | Train Loss: 0.7275 | Val MSE: 28292630.0000 | Val R2: -0.6113
Epoch 069 | Train Loss: 0.7270 | Val MSE: 28418966.0000 | Val R2: -0.6225
Epoch 070 | Train Loss: 0.7276 | Val MSE: 29348374.0000 | Val R2: -0.8250
"""

epochs = []
train_loss = []
val_mse = []
val_r2 = []

for line in log_data.strip().split('\n'):
    match = re.search(r"Epoch (\d+) \| Train Loss: ([\d.]+) \| Val MSE: ([\d.]+) \| Val R2: (-?[\d.]+)", line)
    if match:
        epochs.append(int(match.group(1)))
        train_loss.append(float(match.group(2)))
        val_mse.append(float(match.group(3)))
        val_r2.append(float(match.group(4)))

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss', color='tab:blue')
ax1.plot(epochs, train_loss, color='tab:blue', label='Train Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Val R2', color='tab:red')
ax2.plot(epochs, val_r2, color='tab:red', label='Val R2')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.axhline(0, color='black', linestyle='--', alpha=0.3)

plt.title('Training Performance - ND-full')
fig.tight_layout()
plt.savefig('training_curves.png')
print("Saved training_curves.png")
