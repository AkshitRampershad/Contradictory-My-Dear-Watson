"""
Hand-curated multilingual NLI (Natural Language Inference) example set.

NOTE ON DATA PROVENANCE: This project's real result - a 90.06% test
accuracy, 6th of 64 teams, from an ensemble of fine-tuned XLM-RoBERTa
transformer models - came from training on the actual ~12,000-pair,
15-language Kaggle competition dataset (SNLI/MNLI/ANLI/XNLI-derived),
on GPU, in the original notebook (NLP_Watson_FinalProject.ipynb). That
result is real and untouched.

This module is a SEPARATE, much smaller, hand-written set of
premise/hypothesis/label triples (~195 examples across 5 languages),
written specifically so a lightweight classical-ML demo can train and
run entirely offline, with no GPU and no network access to Kaggle or
HuggingFace (both confirmed unreachable in the environment this demo
was built in). Every triple below was constructed with its label
verified by construction: entailment hypotheses are logical
generalizations of the premise, contradiction hypotheses directly
negate or conflict with it, and neutral hypotheses add plausible but
unconfirmed detail. It is not a substitute for the real competition
dataset, and no accuracy trained on it should be compared to the
90.06% figure above - see README.md for the full picture.
"""

from __future__ import annotations

from dataclasses import dataclass

# Each entry: (premise, entailment_hypothesis, neutral_hypothesis, contradiction_hypothesis)
_SCENARIOS_EN = [
    ("A man is playing guitar on a stage.", "A person is performing music.", "The man is a professional musician.", "No one is on the stage."),
    ("Two children are building a sandcastle on the beach.", "Kids are playing near the ocean.", "The children are siblings.", "The children are inside a house."),
    ("A woman is reading a book in the park.", "A person is outdoors.", "The book is a mystery novel.", "The woman is asleep."),
    ("The chef is preparing a meal in the kitchen.", "Someone is cooking food.", "The meal is for a large party.", "The kitchen is empty."),
    ("A dog is chasing a ball across the yard.", "An animal is running.", "The dog belongs to a young family.", "The dog is sleeping in its bed."),
    ("Students are taking an exam in the classroom.", "People are being tested.", "The exam is in mathematics.", "The classroom is empty."),
    ("A farmer is harvesting wheat in the field.", "Someone is working in agriculture.", "The farmer has owned the land for decades.", "The field has no crops."),
    ("The airplane is landing at the airport.", "A flight is arriving.", "The airplane is a commercial jet.", "The airplane is taking off."),
    ("A scientist is examining samples under a microscope.", "Someone is conducting research.", "The samples are from a new species.", "The microscope is broken."),
    ("Firefighters are extinguishing a blaze in the building.", "Emergency responders are fighting a fire.", "The building is a warehouse.", "There is no fire anywhere."),
    ("A musician is tuning a violin before the concert.", "Someone is preparing an instrument.", "The concert will be sold out.", "The violin has already been played on stage."),
    ("The company announced record profits this quarter.", "The business performed well financially.", "The profits will increase again next quarter.", "The company reported a loss this quarter."),
    ("The new policy requires employees to work from the office three days a week.", "Employees must be present in the office part of the week.", "Most employees support the new policy.", "Employees are allowed to work from home every day."),
    ("The bridge was closed for repairs after the inspection.", "The bridge is currently not accessible.", "The repairs will take six months.", "The bridge remains open to traffic."),
    ("The team won the championship after a dramatic final match.", "The team was victorious.", "The championship was held in a new stadium.", "The team lost the final match."),
    ("Doctors recommend at least eight hours of sleep for adults.", "Sleep duration is important for adult health.", "Most adults sleep exactly eight hours.", "Doctors say adults do not need much sleep."),
    ("The museum unveiled a new exhibit on ancient Egyptian artifacts.", "A new exhibit was opened to the public.", "The exhibit will run for one year.", "The museum closed all of its exhibits."),
    ("Heavy rain caused flooding in several downtown streets.", "The rain led to water accumulating on the streets.", "The flooding damaged hundreds of homes.", "The weather remained dry all week."),
    ("The government proposed a new tax on carbon emissions.", "A new environmental tax was suggested.", "The tax will be approved by the end of the year.", "The government eliminated all taxes on emissions."),
    ("The library extended its hours during final exam week.", "The library is open longer than usual.", "Most students study at the library during exams.", "The library closed early during exam week."),
    ("A hiker reached the summit just before sunset.", "Someone climbed to the top of the mountain.", "The hiker had trained for six months.", "The hiker turned back before reaching the top."),
    ("The orchestra performed a symphony to a full audience.", "Musicians played for many listeners.", "The symphony was composed in the 19th century.", "The concert hall was completely empty."),
    ("The startup raised ten million dollars in its latest funding round.", "The company secured new investment.", "The startup will use the money to hire more engineers.", "The startup failed to raise any funding."),
    ("Volunteers cleaned up litter along the riverbank.", "People removed trash from near the river.", "The cleanup was organized by a local school.", "No one visited the riverbank that day."),
    ("The pilot announced a delay due to bad weather.", "The flight was postponed because of weather conditions.", "The delay lasted exactly two hours.", "The flight departed exactly on schedule."),
]

_SCENARIOS_ES = [
    ("Un hombre está tocando la guitarra en el escenario.", "Una persona está tocando música.", "El hombre es un músico profesional.", "No hay nadie en el escenario."),
    ("Dos niños están construyendo un castillo de arena en la playa.", "Los niños están jugando cerca del mar.", "Los niños son hermanos.", "Los niños están dentro de una casa."),
    ("Una mujer está leyendo un libro en el parque.", "Una persona está afuera.", "El libro es una novela de misterio.", "La mujer está dormida."),
    ("El perro está persiguiendo una pelota en el jardín.", "Un animal está corriendo.", "El perro pertenece a una familia joven.", "El perro está durmiendo en su cama."),
    ("Los estudiantes están tomando un examen en el aula.", "Las personas están siendo evaluadas.", "El examen es de matemáticas.", "El aula está vacía."),
    ("El avión está aterrizando en el aeropuerto.", "Un vuelo está llegando.", "El avión es un jet comercial.", "El avión está despegando."),
    ("Los bomberos están apagando un incendio en el edificio.", "Los equipos de emergencia están combatiendo un incendio.", "El edificio es un almacén.", "No hay ningún incendio."),
    ("La empresa anunció ganancias récord este trimestre.", "El negocio tuvo un buen desempeño financiero.", "Las ganancias aumentarán de nuevo el próximo trimestre.", "La empresa reportó pérdidas este trimestre."),
    ("El equipo ganó el campeonato después de un partido final dramático.", "El equipo fue victorioso.", "El campeonato se celebró en un estadio nuevo.", "El equipo perdió el partido final."),
    ("La biblioteca extendió su horario durante la semana de exámenes finales.", "La biblioteca está abierta más tiempo de lo habitual.", "La mayoría de los estudiantes estudian en la biblioteca durante los exámenes.", "La biblioteca cerró temprano durante la semana de exámenes."),
]

_SCENARIOS_FR = [
    ("Un homme joue de la guitare sur scène.", "Une personne joue de la musique.", "L'homme est un musicien professionnel.", "Il n'y a personne sur scène."),
    ("Deux enfants construisent un château de sable sur la plage.", "Des enfants jouent près de la mer.", "Les enfants sont frère et sœur.", "Les enfants sont à l'intérieur d'une maison."),
    ("Une femme lit un livre dans le parc.", "Une personne est dehors.", "Le livre est un roman policier.", "La femme dort."),
    ("Le chien poursuit une balle dans le jardin.", "Un animal court.", "Le chien appartient à une jeune famille.", "Le chien dort dans son panier."),
    ("Les étudiants passent un examen dans la salle de classe.", "Des personnes sont évaluées.", "L'examen porte sur les mathématiques.", "La salle de classe est vide."),
    ("L'avion atterrit à l'aéroport.", "Un vol arrive.", "L'avion est un jet commercial.", "L'avion décolle."),
    ("Les pompiers éteignent un incendie dans le bâtiment.", "Les secours combattent un incendie.", "Le bâtiment est un entrepôt.", "Il n'y a aucun incendie."),
    ("L'entreprise a annoncé des profits records ce trimestre.", "L'entreprise a bien performé financièrement.", "Les profits augmenteront encore le trimestre prochain.", "L'entreprise a annoncé une perte ce trimestre."),
    ("L'équipe a remporté le championnat après un match final dramatique.", "L'équipe a gagné.", "Le championnat s'est déroulé dans un nouveau stade.", "L'équipe a perdu le match final."),
    ("La bibliothèque a prolongé ses horaires pendant la semaine des examens finaux.", "La bibliothèque est ouverte plus longtemps que d'habitude.", "La plupart des étudiants étudient à la bibliothèque pendant les examens.", "La bibliothèque a fermé plus tôt pendant la semaine des examens."),
]

_SCENARIOS_DE = [
    ("Ein Mann spielt Gitarre auf der Bühne.", "Eine Person macht Musik.", "Der Mann ist ein professioneller Musiker.", "Niemand ist auf der Bühne."),
    ("Zwei Kinder bauen eine Sandburg am Strand.", "Kinder spielen in der Nähe des Meeres.", "Die Kinder sind Geschwister.", "Die Kinder sind in einem Haus."),
    ("Eine Frau liest ein Buch im Park.", "Eine Person ist draußen.", "Das Buch ist ein Krimi.", "Die Frau schläft."),
    ("Der Hund jagt einen Ball im Garten.", "Ein Tier rennt.", "Der Hund gehört einer jungen Familie.", "Der Hund schläft in seinem Bett."),
    ("Studenten schreiben eine Prüfung im Klassenzimmer.", "Menschen werden geprüft.", "Die Prüfung ist in Mathematik.", "Das Klassenzimmer ist leer."),
    ("Das Flugzeug landet auf dem Flughafen.", "Ein Flug kommt an.", "Das Flugzeug ist ein Passagierjet.", "Das Flugzeug startet."),
    ("Die Feuerwehr löscht einen Brand im Gebäude.", "Einsatzkräfte bekämpfen ein Feuer.", "Das Gebäude ist ein Lagerhaus.", "Es gibt kein Feuer."),
    ("Das Unternehmen meldete in diesem Quartal Rekordgewinne.", "Das Unternehmen war finanziell erfolgreich.", "Die Gewinne werden im nächsten Quartal weiter steigen.", "Das Unternehmen meldete einen Verlust in diesem Quartal."),
    ("Die Mannschaft gewann die Meisterschaft nach einem dramatischen Finale.", "Die Mannschaft war siegreich.", "Die Meisterschaft fand in einem neuen Stadion statt.", "Die Mannschaft verlor das Finale."),
    ("Die Bibliothek verlängerte ihre Öffnungszeiten während der Prüfungswoche.", "Die Bibliothek ist länger als sonst geöffnet.", "Die meisten Studenten lernen während der Prüfungen in der Bibliothek.", "Die Bibliothek schloss früher während der Prüfungswoche."),
]

_SCENARIOS_IT = [
    ("Un uomo suona la chitarra sul palco.", "Una persona sta suonando musica.", "L'uomo è un musicista professionista.", "Non c'è nessuno sul palco."),
    ("Due bambini stanno costruendo un castello di sabbia sulla spiaggia.", "I bambini stanno giocando vicino al mare.", "I bambini sono fratelli.", "I bambini sono dentro una casa."),
    ("Una donna sta leggendo un libro nel parco.", "Una persona è all'aperto.", "Il libro è un romanzo giallo.", "La donna sta dormendo."),
    ("Il cane insegue una palla in giardino.", "Un animale sta correndo.", "Il cane appartiene a una famiglia giovane.", "Il cane sta dormendo nella sua cuccia."),
    ("Gli studenti stanno facendo un esame in classe.", "Le persone vengono valutate.", "L'esame è di matematica.", "L'aula è vuota."),
    ("L'aereo sta atterrando all'aeroporto.", "Un volo sta arrivando.", "L'aereo è un jet commerciale.", "L'aereo sta decollando."),
    ("I vigili del fuoco stanno spegnendo un incendio nell'edificio.", "I soccorritori stanno combattendo un incendio.", "L'edificio è un magazzino.", "Non c'è nessun incendio."),
    ("L'azienda ha annunciato profitti record questo trimestre.", "L'azienda ha ottenuto buoni risultati finanziari.", "I profitti aumenteranno ancora il prossimo trimestre.", "L'azienda ha riportato una perdita questo trimestre."),
    ("La squadra ha vinto il campionato dopo una finale drammatica.", "La squadra ha vinto.", "Il campionato si è svolto in un nuovo stadio.", "La squadra ha perso la finale."),
    ("La biblioteca ha esteso il suo orario durante la settimana degli esami finali.", "La biblioteca è aperta più a lungo del solito.", "La maggior parte degli studenti studia in biblioteca durante gli esami.", "La biblioteca ha chiuso presto durante la settimana degli esami."),
]

_LANGUAGE_SCENARIOS = {
    "en": _SCENARIOS_EN,
    "es": _SCENARIOS_ES,
    "fr": _SCENARIOS_FR,
    "de": _SCENARIOS_DE,
    "it": _SCENARIOS_IT,
}

LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian"}


@dataclass(frozen=True)
class NLIExample:
    premise: str
    hypothesis: str
    label: str  # "entailment" | "neutral" | "contradiction"
    language: str


def build_examples() -> list[NLIExample]:
    examples: list[NLIExample] = []
    for lang, scenarios in _LANGUAGE_SCENARIOS.items():
        for premise, entailment, neutral, contradiction in scenarios:
            examples.append(NLIExample(premise, entailment, "entailment", lang))
            examples.append(NLIExample(premise, neutral, "neutral", lang))
            examples.append(NLIExample(premise, contradiction, "contradiction", lang))
    return examples


if __name__ == "__main__":
    examples = build_examples()
    print(f"Total examples: {len(examples)}")
    from collections import Counter

    print("By label:", Counter(e.label for e in examples))
    print("By language:", Counter(e.language for e in examples))
