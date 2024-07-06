
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd

# Configurarea modelului GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Colectarea datelor
data = {
    "client_request": [
        "Dezvoltarea unei aplicații de tip Glovo cu următoarele secțiuni: Sistem de logistică a rider-ului, Înregistrarea restaurantelor în aplicație, Primirea comenzilor de către clienți, Posibilitatea de a adăuga sau elimina produse din meniul restaurantului, Plata cu cardul."
    ],
    "offer_example": [
        """
        Descrierea aplicației solicitate:
        Aplicația va fi o platformă de livrare de mâncare similară cu Glovo, care include următoarele secțiuni:
        
        1. Sistem de logistică a rider-ului:
            - Monitorizarea și gestionarea livrărilor.
            - Atribuirea comenzilor către rideri în funcție de locație și disponibilitate.
        
        2. Înregistrarea restaurantelor în aplicație:
            - Formular de înscriere pentru restaurante.
            - Validarea și aprobarea restaurantelor de către administratori.
        
        3. Primirea comenzilor de către clienți:
            - Interfață intuitivă pentru plasarea comenzilor.
            - Notificări în timp real pentru starea comenzii.
        
        4. Posibilitatea de a adăuga sau elimina produse din meniul restaurantului:
            - Dashboard pentru administrarea meniului de către restaurante.
            - Actualizări în timp real ale meniului.
        
        5. Plata cu cardul:
            - Integrarea cu procesatori de plăți pentru tranzacții sigure.
            - Opțiuni multiple de plată (card de credit, debit).
        
        Task-uri Adiționale Ne-meneționate de Client, dar Necesare:
        - Secțiunea financiară: cum restaurantele solicită bani de la administratorii aplicației.
        - Modalități de plată pentru rideri.
        - Generarea automată a facturilor și posibilitatea clientului de a descărca factura generată.
        """
    ]
}

df = pd.DataFrame(data)

class CustomDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {
            'input_ids': tokenizer(item['client_request'], return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.flatten(),
            'labels': tokenizer(item['offer_example'], return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids.flatten()
        }

train_dataset = CustomDataset(df)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

def generate_offer(client_request):
    inputs = tokenizer(client_request, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(inputs["input_ids"], max_length=500, pad_token_id=tokenizer.eos_token_id)
    offer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return offer

# Exemplu de solicitare de la client
client_request = "Dezvoltarea unei aplicații de tip Glovo cu următoarele secțiuni: Sistem de logistică a rider-ului, Înregistrarea restaurantelor în aplicație, Primirea comenzilor de către clienți, Posibilitatea de a adăuga sau elimina produse din meniul restaurantului, Plata cu cardul."

generated_offer = generate_offer(client_request)
print("Oferta generată:")
print(generated_offer)

# Validarea ofertei
def validate_offer(generated_offer):
    # Aceasta parte poate fi extinsă pentru a include validări specifice
    print("Validarea ofertei:")
    print(generated_offer)

validate_offer(generated_offer)