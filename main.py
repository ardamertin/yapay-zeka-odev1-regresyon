import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 12
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

p = np.linspace(-2, 2, 21).reshape(-1, 1).astype(np.float32)
y = (1.0 + np.sin(np.pi * p / 4.0)).astype(np.float32)

p_dense = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)
y_dense = 1.0 + np.sin(np.pi * p_dense / 4.0)

os.makedirs("grafikler", exist_ok=True)

def get_initializer(dist_name, seed=SEED):
    if dist_name == 'normal':
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=seed)
    elif dist_name == 'uniform':
        return tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=seed)
    return 'glorot_uniform'

def get_optimizer(opt_name, lr):
    if opt_name == 'SGD':
        return tf.keras.optimizers.SGD(learning_rate=lr)
    elif opt_name == 'RMSProp':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == 'ADAM':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    return tf.keras.optimizers.SGD(learning_rate=lr)

def build_and_train_model(act='relu', lr=0.1, opt='SGD', dist='normal', n_layers=1, n_neurons=2, epochs=300):
    tf.keras.utils.set_random_seed(SEED)
    init = get_initializer(dist, SEED)
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(1,)))
    
    for _ in range(n_layers):
        if act == 'leaky_relu':
            model.add(layers.Dense(n_neurons, kernel_initializer=init))
            model.add(layers.LeakyReLU(negative_slope=0.2))
        elif act == 'silu':
            model.add(layers.Dense(n_neurons, activation='swish', kernel_initializer=init))
        else:
            model.add(layers.Dense(n_neurons, activation=act, kernel_initializer=init))
            
    model.add(layers.Dense(1, activation='linear', kernel_initializer=init))
    
    optimizer = get_optimizer(opt, lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    p_tf = tf.constant(p)
    y_tf = tf.constant(y)
    ss_tot = float(np.sum((y - np.mean(y))**2))
    
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            preds = model(p_tf, training=True)
            loss = loss_fn(y_tf, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, preds

    losses = []
    r2_history = []
    
    t0 = time.time()
    for _ in range(epochs):
        l, step_preds = train_step()
        l_val = float(l)
        losses.append(l_val)
        ss_res = float(np.sum((y - step_preds.numpy())**2))
        r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r2_history.append(r2_val)
    elapsed = time.time() - t0
    
    preds = model(p_tf, training=False).numpy()
    mse = float(np.mean((y - preds)**2))
    rmse = float(np.sqrt(mse))
    final_r2 = r2_history[-1]
    preds_dense = model(tf.constant(p_dense), training=False).numpy()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': final_r2,
        'time': elapsed,
        'losses': losses,
        'r2_history': r2_history,
        'preds': preds,
        'preds_dense': preds_dense
    }

def print_table(title, results, best_key):
    print("\n" + "="*75)
    print(f" {title.upper()} ")
    print("="*75)
    print(f"{'Parametre Değeri':<25} | {'MSE':<12} | {'RMSE':<12} | {'R2':<12} | {'Süre (sn)':<10}")
    print("-" * 75)
    for k, v in results.items():
        prefix = "-> [EN IYI] " if k == best_key else "   "
        row_str = f"{prefix + str(k):<25} | {v['mse']:<12.6f} | {v['rmse']:<12.6f} | {v['r2']:<12.6f} | {v['time']:<10.3f}"
        print(row_str)
    print("="*75)

def save_best_plot(filename, title, sim_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.03)
    
    axes[0].plot(sim_data['losses'], color='#c0392b', lw=2)
    axes[0].set_yscale('log')
    axes[0].set_title("Epoch / MSE Kayıp (Logaritmik)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("MSE Kayıp (log)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(p_dense, y_dense, label='Hedef: $y = 1 + \\sin(\\pi p / 4)$', color='#2c3e50', lw=2.5)
    axes[1].scatter(p, y, color='#2980b9', s=35, zorder=5, label='Eğitim Noktaları')
    axes[1].plot(p_dense, sim_data['preds_dense'], label='Model Çıkışı', color='#e74c3c', lw=2, linestyle='--')
    axes[1].set_title(f"Hedef vs Ağ Çıkışı ($R^2={sim_data['r2']:.4f}$)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("p (Giriş)", fontsize=10)
    axes[1].set_ylabel("y (Çıkış)", fontsize=10)
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[2].plot(sim_data['r2_history'], color='#27ae60', lw=2)
    axes[2].set_title("Epoch / $R^2$ Skoru Değişimi", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("$R^2$ Skoru", fontsize=10)
    axes[2].set_ylim([-0.1, 1.05])
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    acts = [('Lineer', 'linear'), ('Sigmoid', 'sigmoid'), ('Hiperbolik Tanjant', 'tanh'),
            ('ReLU', 'relu'), ('LeakyReLU', 'leaky_relu'), ('ELU', 'elu'), ('SİLU', 'silu')]
    res_tab1 = {name: build_and_train_model(act=k) for name, k in acts}
    best_tab1 = min(res_tab1.keys(), key=lambda k: res_tab1[k]['mse'])
    print_table("Tablo 1: Aktivasyon Fonksiyonlarının Değişimi", res_tab1, best_tab1)
    save_best_plot("grafikler/tablo1_en_iyi.png", f"Tablo 1 En İyi: {best_tab1}", res_tab1[best_tab1])
    
    lrs = [0.1, 0.01, 0.001]
    res_tab2 = {str(lr): build_and_train_model(lr=lr) for lr in lrs}
    best_tab2 = min(res_tab2.keys(), key=lambda k: res_tab2[k]['mse'])
    print_table("Tablo 2: Öğrenme Oranının Değişimi", res_tab2, best_tab2)
    save_best_plot("grafikler/tablo2_en_iyi.png", f"Tablo 2 En İyi: lr = {best_tab2}", res_tab2[best_tab2])
    
    opts = ['SGD', 'RMSProp', 'ADAM']
    res_tab3 = {opt: build_and_train_model(opt=opt) for opt in opts}
    best_tab3 = min(res_tab3.keys(), key=lambda k: res_tab3[k]['mse'])
    print_table("Tablo 3: Öğrenme Oranı Ayar Yöntemi (Optimizer) Değişimi", res_tab3, best_tab3)
    save_best_plot("grafikler/tablo3_en_iyi.png", f"Tablo 3 En İyi: {best_tab3}", res_tab3[best_tab3])
    
    dists = [('Normal', 'normal'), ('Uniform', 'uniform')]
    res_tab4 = {name: build_and_train_model(dist=d) for name, d in dists}
    best_tab4 = min(res_tab4.keys(), key=lambda k: res_tab4[k]['mse'])
    print_table("Tablo 4: Ağırlık Başlangıç Dağılımı Değişimi", res_tab4, best_tab4)
    save_best_plot("grafikler/tablo4_en_iyi.png", f"Tablo 4 En İyi: {best_tab4}", res_tab4[best_tab4])
    
    layers_list = [1, 2, 5, 10, 20]
    res_tab5 = {str(l): build_and_train_model(n_layers=l) for l in layers_list}
    best_tab5 = min(res_tab5.keys(), key=lambda k: res_tab5[k]['mse'])
    print_table("Tablo 5: Gizli Katman Sayısının Değişimi", res_tab5, best_tab5)
    save_best_plot("grafikler/tablo5_en_iyi.png", f"Tablo 5 En İyi: {best_tab5} Katman", res_tab5[best_tab5])
    
    neurons_list = [2, 5, 10, 20, 50]
    res_tab6 = {str(n): build_and_train_model(n_neurons=n) for n in neurons_list}
    best_tab6 = min(res_tab6.keys(), key=lambda k: res_tab6[k]['mse'])
    print_table("Tablo 6: Gizli Katmandaki Nöron Sayısının Değişimi", res_tab6, best_tab6)
    save_best_plot("grafikler/tablo6_en_iyi.png", f"Tablo 6 En İyi: {best_tab6} Nöron", res_tab6[best_tab6])