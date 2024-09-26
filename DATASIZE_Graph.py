import pandas as pd
import matplotlib.pyplot as plt

# Charger les données du fichier CSV
df = pd.read_csv('results_summary.csv')

# Filtrer les données pour ne garder que celles où 'Dataset' est 'Test'
df_test = df[df['Dataset'] == 'Test']

# Calculer la moyenne et l'écart type de 'Syllable Error Rate' pour chaque valeur unique de 'Sequences' dans le dataset Test
mean_phrase_error_rate_test = df_test.groupby('Sequences')['Syllable Error Rate'].mean()
std_phrase_error_rate_test = df_test.groupby('Sequences')['Syllable Error Rate'].std()

# Générer le graphique
plt.figure(figsize=(10, 6))

# Tracer la moyenne
plt.plot(mean_phrase_error_rate_test.index, mean_phrase_error_rate_test.values, marker='o', label='Mean')

# Remplir la zone de l'écart type
plt.fill_between(mean_phrase_error_rate_test.index,
                 mean_phrase_error_rate_test.values - std_phrase_error_rate_test.values,
                 mean_phrase_error_rate_test.values + std_phrase_error_rate_test.values,
                 color='blue', alpha=0.2, label='Std_dev')

# Ajouter les labels et un titre
plt.xlabel('Number of Sequences')
plt.ylabel('Average Phrase Error Rate (%)')
plt.title('Phrase Error Rate of Canapy with Marron1')

# Ajuster les ticks de l'axe des abscisses pour correspondre exactement aux séquences
plt.xticks(mean_phrase_error_rate_test.index)  # Utiliser uniquement les séquences présentes

# Afficher la grille
plt.grid(True)

# Ajouter la légende
plt.legend()

# Enregistrer le graphique au format PNG (ou autre format selon les besoins)
plt.savefig('phrase_error_rate_graph.png', format='png', dpi=300)  # DPI = résolution

# Afficher le graphique
plt.show()
