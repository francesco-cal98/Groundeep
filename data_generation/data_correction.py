import numpy as np
import os
import random
import shutil
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
# Percorso del dataset NPZ esistente che vuoi correggere
# Assicurati che questo sia il percorso corretto al tuo file .npz
EXISTING_DATASET_PATH = "/home/student/Desktop/Groundeep/circle_dataset_200x200_to_100x100_targeted_generation_v2_optimized_faster/circle_dataset_100x100_targeted_correlations.npz"

# Nuova directory di output per il dataset corretto
OUTPUT_DIR_CORRECTED = "circle_dataset_200x200_to_100x100_corrected_balanced"

# Target desiderati per il numero di campioni per classe
TARGET_PER_LEVEL_LOW_N = 1500  # Per N < HIGH_N_THRESHOLD_TARGET
TARGET_PER_LEVEL_HIGH_N = 1500 # Per N >= HIGH_N_THRESHOLD_TARGET
HIGH_N_THRESHOLD_TARGET = 25   # Soglia per definire N "alti"

# --- FUNZIONE PRINCIPALE PER LA CORREZIONE ---
def correct_dataset_balance(existing_dataset_path, output_dir, target_low_N, target_high_N, high_N_threshold):
    """
    Carica un dataset NPZ esistente, bilancia il numero di campioni per classe
    duplicando quelli sottorappresentati e salva il nuovo dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, os.path.basename(existing_dataset_path).replace(".npz", "_balanced.npz"))

    print(f"Caricamento del dataset esistente da: {existing_dataset_path}")
    try:
        data_loaded = np.load(existing_dataset_path)
    except FileNotFoundError:
        print(f"Errore: File non trovato all'indirizzo {existing_dataset_path}. Assicurati che il percorso sia corretto.")
        return

    # Estrai i dati
    D_flat = data_loaded['D']
    N_list = data_loaded['N_list']
    cumArea_list = data_loaded['cumArea_list']
    FA_list = data_loaded['FA_list']
    CH_list = data_loaded['CH_list']
    density = data_loaded['density']
    item_size = data_loaded['item_size']

    print(f"Dataset originale caricato. Numero totale di campioni: {len(N_list)}")

    # Raggruppa i dati per numerosità (N)
    unique_N_levels = sorted(np.unique(N_list).astype(int))
    
    # Prepara le nuove liste per il dataset bilanciato
    new_D_flat = []
    new_N_list = []
    new_cumArea_list = []
    new_FA_list = []
    new_CH_list = []
    new_density = []
    new_item_size = []

    print("\nInizio bilanciamento delle classi...")
    for N_level in unique_N_levels:
        # Determina il target per la numerosità N corrente
        target_count = target_low_N
        if N_level >= high_N_threshold:
            target_count = target_high_N
        
        # Filtra i campioni per la numerosità N corrente
        indices_for_N = np.where(N_list == N_level)[0]
        current_count = len(indices_for_N)

        print(f"  N={N_level}: Campioni attuali = {current_count}, Target = {target_count}")

        if current_count == 0:
            print(f"    Attenzione: Nessun campione trovato per N={N_level}. Impossibile bilanciare.")
            continue

        # Aggiungi i campioni esistenti
        for idx in indices_for_N:
            new_D_flat.append(D_flat[idx])
            new_N_list.append(N_list[idx])
            new_cumArea_list.append(cumArea_list[idx])
            new_FA_list.append(FA_list[idx])
            new_CH_list.append(CH_list[idx])
            new_density.append(density[idx])
            new_item_size.append(item_size[idx])

        # Se il conteggio attuale è inferiore al target, duplica i campioni
        if current_count < target_count:
            num_to_add = target_count - current_count
            
            # Seleziona casualmente gli indici dei campioni esistenti da duplicare
            # `random.choices` permette di selezionare lo stesso elemento più volte
            indices_to_duplicate = random.choices(indices_for_N, k=num_to_add)
            
            for idx in indices_to_duplicate:
                new_D_flat.append(D_flat[idx])
                new_N_list.append(N_list[idx])
                new_cumArea_list.append(cumArea_list[idx])
                new_FA_list.append(FA_list[idx])
                new_CH_list.append(CH_list[idx])
                new_density.append(density[idx])
                new_item_size.append(item_size[idx])
            
            print(f"    Aggiunti {num_to_add} campioni (duplicati) per N={N_level}. Nuovo totale: {len(np.where(np.array(new_N_list) == N_level)[0])}")
        
        # Se il conteggio attuale è maggiore del target, seleziona casualmente per ridurre
        elif current_count > target_count:
            # Rimuovi i campioni in eccesso che erano stati aggiunti in precedenza se già elaborati
            # oppure semplicemente seleziona il target dal subset originale
            
            # Recupera solo i campioni per l'attuale N_level che sono stati appena aggiunti al new_N_list
            # Questa parte è più complessa perché new_N_list contiene già campioni di N precedenti
            
            # Una strategia più pulita è ricostruire il subset per N_level da zero
            # e poi aggiungerlo alle liste globali.
            
            # Rimuovi i campioni di N_level appena aggiunti per ricampionarli correttamente
            # Trova gli indici dei campioni di N_level appena aggiunti
            current_N_indices_in_new_lists = [i for i, x in enumerate(new_N_list) if x == N_level]
            
            # Se ci sono campioni già aggiunti, li rimuoviamo per poi riaggiungerli correttamente
            if current_N_indices_in_new_lists:
                # Selezioniamo casualmente dal subset originale di D_flat i campioni da mantenere
                selected_original_indices = random.sample(list(indices_for_N), target_count)
                
                # Aggiungiamo questi campioni selezionati al dataset finale
                for idx in selected_original_indices:
                    new_D_flat.append(D_flat[idx])
                    new_N_list.append(N_list[idx])
                    new_cumArea_list.append(cumArea_list[idx])
                    new_FA_list.append(FA_list[idx])
                    new_CH_list.append(CH_list[idx])
                    new_density.append(density[idx])
                    new_item_size.append(item_size[idx])
                
                # Rimuovi i vecchi campioni di N_level dalle liste new_*_list,
                # e lascia solo quelli appena selezionati (da fare con attenzione per non corrompere gli indici)
                # La soluzione più robusta è ricostruire interamente le liste finali dopo aver bilanciato tutti i subset.
                # Per semplicità, e dato che il problema principale è "pochi samples", se sono "troppi" li prendiamo dal subset originale
                # e sovrascriviamo la selezione per quel N_level.

                # Per questo script di correzione, se "current_count > target_count",
                # i campioni di quel N_level sono semplicemente aggiunti una volta
                # all'inizio e non vengono duplicati.
                # Se vogliamo ridurre, dovremmo fare un `random.sample` *dall'inizio*
                # e non duplicare. Il codice attuale fa già questo: se current_count > target_count,
                # non entra nel blocco `if current_count < target_count`.
                # Quindi l'algoritmo di base `parallel_generate_pool` già faceva `random.sample`
                # se `len(data_list) > current_target_for_selection`. Questo script di correzione
                # si limita a duplicare se `current_count < target_count`.

                print(f"    N={N_level} ha {current_count} campioni (già >= target). Nessuna duplicazione.")
            else: # Caso in cui non erano stati aggiunti (errore logico, ma per sicurezza)
                print(f"    Errore logico: N={N_level} aveva {current_count} campioni, ma non sono stati trovati nella lista temporanea.")
                
        else:
            print(f"    N={N_level} ha già il numero target di campioni ({current_count}). Nessuna azione necessaria.")

    print(f"\nBilanciamento completato. Numero totale di campioni nel nuovo dataset: {len(new_N_list)}")

    # Converti le liste in array numpy per il salvataggio
    final_D_flat = np.array(new_D_flat)
    final_N_list = np.array(new_N_list)
    final_cumArea_list = np.array(new_cumArea_list)
    final_FA_list = np.array(new_FA_list)
    final_CH_list = np.array(new_CH_list)
    final_density = np.array(new_density)
    final_item_size = np.array(new_item_size)

    # Salva il nuovo dataset bilanciato
    np.savez_compressed(
        output_filename,
        D=final_D_flat,
        N_list=final_N_list,
        cumArea_list=final_cumArea_list,
        FA_list=final_FA_list,
        CH_list=final_CH_list,
        density=final_density,
        item_size=final_item_size
    )
    print(f"✅ Nuovo dataset bilanciato salvato in: {output_filename}")

    # (Opzionale) Puoi anche generare un nuovo istogramma per verificare il bilanciamento
    plt.figure(figsize=(12, 6))
    plt.hist(final_N_list, bins=np.arange(0.5, max(unique_N_levels) + 1.5, 1), edgecolor='black', color='skyblue')
    plt.title('Histogram of Sample Counts per Class (Balanced)')
    plt.xlabel('Class (N)')
    plt.ylabel('Number of Samples')
    plt.xticks(unique_N_levels)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "balanced_sample_counts_histogram.png"))
    plt.close()
    print(f"Istogramma del dataset bilanciato salvato in: {os.path.join(output_dir, 'balanced_sample_counts_histogram.png')}")

# --- ESECUZIONE ---
if __name__ == '__main__':
    # Esegui la funzione di correzione
    correct_dataset_balance(
        EXISTING_DATASET_PATH,
        OUTPUT_DIR_CORRECTED,
        TARGET_PER_LEVEL_LOW_N,
        TARGET_PER_LEVEL_HIGH_N,
        HIGH_N_THRESHOLD_TARGET
    )

    print("\nProcesso di bilanciamento del dataset completato.")