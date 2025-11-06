# 🤝 Human-in-the-Loop (HITL) Lietošanas Instrukcija

## 📋 Saturs

- [Par HITL](#par-hitl)
- [Kāpēc HITL ir svarīgs](#kāpēc-hitl-ir-svarīgs)
- [HITL Workflow](#hitl-workflow)
- [Lietošanas Piemēri](#lietošanas-piemēri)
- [Best Practices](#best-practices)

## 🎯 Par HITL

**Human-in-the-Loop (HITL)** ir sistēmas galvenā funkcija, kas ļauj cilvēkam:

1. **Pārskatīt** AI aģentu analīzi PIRMS izmaiņu piemērošanas
2. **Apstiprināt vai noraidīt** katru ieteikto izmaiņu
3. **Modificēt** AI ieteikumus pēc saviem ieskatiem
4. **Mācīt sistēmu** caur feedback

## ❓ Kāpēc HITL ir svarīgs?

### Priekšrocības:

✅ **Kontrole**: Jūs kontrolējat VISAS izmaiņas savos datos
✅ **Drošība**: Nekad nenotiek nevēlamas vai kaitīgas transformācijas
✅ **Uzticamība**: Pilna redzamība, ko sistēma darīs
✅ **Mācīšanās**: Sistēma kļūst labāka no jūsu lēmumiem
✅ **Atbilstība**: Audit trail regulatīvajām prasībām
✅ **Domain Expertise**: Jūsu nozares zināšanas uzlabo AI

### Bez HITL:

❌ AI var pieņemt nepareizus lēmumus
❌ Dati var tikt bojāti
❌ Nevēlamas izmaiņas bez jūsu ziņas
❌ Grūti izskaidrot lēmumus

## 🔄 HITL Workflow

### Stage 1: 📤 Datu Augšupielāde

```
1. Izvēlieties failu (CSV, Excel, JSON, Parquet)
2. Sistēma ielādē datus
3. Redzat tūlītēju priekšskatījumu:
   - Rindu un kolonnu skaits
   - Trūkstošo vērtību %
   - Dublikātu skaits
```

**Jūsu darbība:** Pārliecinaties, ka dati pareizi ielādējušies.

---

### Stage 2: 📊 Novērtējums (Assessment)

```
1. Izvēlaties kvalitātes dimensijas:
   ☑️ Precision (precizitāte)
   ☑️ Completeness (pilnīgums)
   ☑️ Reusability (atkārtota izmantošana)

2. Klikšķiniet "Sākt Analīzi"

3. AI aģenti analizē datus:
   🤖 Precision Agent → format issues, outliers
   🤖 Completeness Agent → missing values
   🤖 Reusability Agent → FAIR compliance
```

**Jūsu darbība:** Nogaidiet analīzes pabeigšanu (~30-60 sekundes).

---

### Stage 3: 🔍 Pārskatīšana (Review) - SVARĪGĀKAIS!

```
📊 Kopējais Kvalitātes Vērtējums
├─ Rezultāts: 67.3% 🟡
├─ Konstatētas problēmas: 15
└─ AI pārliecība: 85%

📈 Kvalitātes Dimensijas
├─ Precision: 72.1%     (5 problēmas)
├─ Completeness: 58.2%  (8 problēmas)
└─ Reusability: 71.5%   (2 problēmas)

🔍 Konstatētās Problēmas
├─ 🔴 CRITICAL: 2
│   └─ Missing required field: patient_id (3 rows)
├─ 🟠 HIGH: 5
│   ├─ Mixed types in 'age' column
│   └─ Outliers detected in 'salary' (15 values)
├─ 🟡 MEDIUM: 6
└─ ⚪ LOW: 2

💡 Ieteiktie Uzlabojumi
┌──────────────────────────────────────────────┐
│ Nr. │ Darbība              │ Kolonna │ Svarīgums │
├─────┼──────────────────────┼─────────┼───────────┤
│  1  │ impute_missing_values│ age     │ critical  │
│  2  │ handle_outliers      │ salary  │ high      │
│  3  │ standardize_types    │ age     │ high      │
└──────────────────────────────────────────────┘
```

**Jūsu darbība:**
- Pārskatiet VISAS problēmas
- Izlemiet, vai piekrītat AI analīzei
- Sagatavojieties apstiprināšanai

**❗ SVARĪGI:** Nekādas izmaiņas vēl NAV piemērotas!

---

### Stage 4: ✅ Apstiprināšana (Approval) - JŪSU LĒMUMS!

```
╔══════════════════════════════════════════════╗
║           Izmaiņa #1                          ║
╠══════════════════════════════════════════════╣
║ Darbība:  impute_missing_values              ║
║ Mērķis:   age                                 ║
║ Apraksts: Aizpildīt trūkstošās vērtības      ║
║           kolonnā 'age'                       ║
║ Svarīgums: critical                          ║
║ Ietekme:  Ļoti liela ietekme - ieteicams    ║
║           apstiprināt                         ║
╠══════════════════════════════════════════════╣
║ [✅ Apstiprināt]  [❌ Noraidīt]              ║
╚══════════════════════════════════════════════╝
```

**Jūs varat:**

1. **✅ Apstiprināt** - izmaiņa tiks piemērota
2. **❌ Noraidīt** - izmaiņa NETIKS piemērota
3. **Masveida darbības:**
   - "✅ Apstiprināt Visas" - ja uzticaties AI
   - "❌ Noraidīt Visas" - ja nevēlaties izmaiņas

**Piemērs - Kad apstiprināt:**
```
✅ "Aizpildīt trūkstošās vērtības 'age' ar median"
   → Labi, ja zināt, ka age ir svarīgs un median ir piemērots

✅ "Noņemt outliers no 'salary'"
   → Labi, ja redzat, ka outliers ir kļūda

✅ "Standartizēt kolonnu nosaukumus"
   → Drošs, neliks bojā datus
```

**Piemērs - Kad noraidīt:**
```
❌ "Aizpildīt trūkstošās vērtības 'diagnosis' ar mode"
   → NAV labi, jo diagnosis ir pārāk specifisks

❌ "Noņemt outliers no 'blood_pressure'"
   → Varbūt tie nav outliers, bet reāli simptomi

❌ "Dzēst kolonnu 'notes'"
   → Varbūt notes ir svarīgi
```

**Jūsu darbība:** Pārskatiet un pieņemiet lēmumu par KATRU izmaiņu!

---

### Stage 5: 🔄 Transformācija

```
🔄 Piemēro izmaiņas...

✅ Apstiprinātās izmaiņas: 8
❌ Noraidītās izmaiņas: 4

Progress: ████████████████░░░░ 80%

Piemēro:
✅ impute_missing_values → age (median)
✅ handle_outliers → salary (clip)
✅ standardize_types → age (numeric)
❌ SKIP: remove_column → notes (rejected)
...
```

**Jūsu darbība:** Nogaidiet, kamēr sistēma piemēro TIKAI apstiprinātās izmaiņas.

---

### Stage 6: 📈 Rezultāti

```
✅ Datu kvalitāte uzlabota!

📊 Salīdzinājums: Pirms ↔️ Pēc
┌─────────────────┬─────────┬─────────┐
│                 │  Pirms  │  Pēc    │
├─────────────────┼─────────┼─────────┤
│ Rindas          │  1000   │  1000   │
│ Kolonnas        │   25    │   25    │
│ Trūkstošas      │  342    │   28    │ ↓-314
│ Kvalitāte       │  67%    │   92%   │ ↑+25%
└─────────────────┴─────────┴─────────┘

💾 Lejupielāde
[📥 Lejupielādēt CSV]

🔄 [Sākt No Jauna]
```

**Jūsu darbība:**
- Pārskatiet uzlabojumus
- Lejupielādējiet uzlabotus datus
- Sāciet no jauna ar citiem datiem

---

## 💡 Lietošanas Piemēri

### Piemērs 1: Medicīnas dati ar trūkstošām vērtībām

**Scenārijs:** Pacientu dati ar trūkstošām 'age' vērtībām.

```python
# Dati
patient_id | age  | diagnosis
1001       | 45   | diabetes
1002       | None | hypertension
1003       | 67   | diabetes
```

**AI ieteikums:**
```
💡 Aizpildīt trūkstošās vērtības 'age' ar median (56)
```

**Jūsu lēmums:**
```
✅ APSTIPRINĀT, ja:
   - age ir svarīgs jūsu analīzē
   - median ir piemērots jūsu datiem
   - zināt, ka trūkstošās vērtības ir nejauši

❌ NORAIDĪT, ja:
   - age NAV svarīgs
   - vēlaties tos dzēst pilnībā
   - trūkstošās vērtības nav nejauši
```

---

### Piemērs 2: Outliers finanšu datos

**Scenārijs:** Algu dati ar iespējamiem outliers.

```python
# Dati
employee_id | salary
E001        | 45000
E002        | 52000
E003        | 999999  # Outlier?
E004        | 48000
```

**AI ieteikums:**
```
💡 Noņemt outliers no 'salary' (clip pie 150000)
```

**Jūsu lēmums:**
```
✅ APSTIPRINĀT, ja:
   - 999999 ir acīmredzama kļūda
   - zināt algu diapazonu

❌ NORAIDĪT, ja:
   - tā var būt CEO alga (valida)
   - vēlaties to pārbaudīt manuāli
```

---

## 🎯 Best Practices

### 1. Vienmēr Pārskatiet Review Stage

❌ SLIKTI:
```
Augšupielādēt → Analīze → [✅ Apstiprināt Visas] → Gatavs
```

✅ LABI:
```
Augšupielādēt → Analīze → PĀRSKATĪT REZULTĀTUS →
Izlemt katru izmaiņu → Piemērot → Pārbaudīt rezultātus
```

---

### 2. Sapratiet, KO izmaiņa darīs

Pirms apstiprināšanas, jautājiet sev:
- Vai es saprotu, ko šī darbība darīs?
- Vai tas ir piemērots maniem datiem?
- Vai rezultāts būs tas, ko es vēlos?

---

### 3. Sāciet ar Mazām Partijām

Ja nezināt sistēmu:
1. Sāciet ar 100-1000 rindām
2. Pārskatiet rezultātus
3. Mācieties, kā AI pieņem lēmumus
4. Tad lietojiet uz lielākiem datiem

---

### 4. Dokumentējiet Savu Lēmumu

Ja sistēma prasa komentāru, paskaidrojiet:
```
✅ Apstiprināts: "Median ir piemērots šim vecuma diapazonam"
❌ Noraidīts: "Diagnosis nedrīkst automātiski aizpildīt"
```

Tas palīdz:
- Jums atcerēties vēlāk
- Sistēmai mācīties
- Komandai saprast lēmumus

---

### 5. Pārbaudiet Rezultātus

Pēc transformācijas:
1. Skatieties salīdzinājumu
2. Lejupielādējiet datus
3. Pārbaudiet dažas rindas manuāli
4. Убедитесь, ka viss ir kā gaidīts

---

## ⚠️ Brīdinājumi

### ❗ Nekad neapstiprināt, ja nezināt

Ja nesaprotat, ko darbība dara:
1. Noraidiet to
2. Pajautājiet kolēģim/ekspertam
3. Izlasiet dokumentāciju
4. Tikai tad apstipriniet

### ❗ Kritiskiem datiem - Extra uzmanība

Ja dati ir kritiski (medicīna, finanses, juridisks):
- Pārskatiet KATRU izmaiņu
- Dokumentējiet VISUS lēmumus
- Uzglabājiet oriģinālos datus
- Konsultējieties ar domain ekspertiem

### ❗ Backup ir obligāts

Pirms jebkādas apstrādes:
1. Izveidojiet backup
2. Saglabājiet oriģinālos datus
3. Dokumentējiet, kas tika mainīts

---

## 📞 Palīdzība

Ja jums ir jautājumi:
- GitHub Issues: [Link]
- Documentation: `docs/`
- Email: [kontakts]

---

**Atcerieties:** HITL ir tur, lai JUMS būtu kontrole. Uzticieties saviem instinktiem un domain zināšanām!
