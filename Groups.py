from random import randint

teilnehmer = ["Eric", "Paerli", "Gaebu", "Maettu", "Vaettu", "Roemu", "Stefu"]
zutaten = ["Wiederchoier", "Auesfraesser", "beschnablets Tier"]
alk = ["Rum", "Vodka", "Gin"]
samstag = ["Vaettu", "", ""]
freitag = ["Vaettu", "Roemu"]
donnerstag = ["Vaettu", "Roemu"]

print("Zuteilungen: ")

def constraints():
    fri = ((freitag[0] == "Vaettu" and freitag[1] == "Roemu") or (freitag[1] == "Vaettu" and freitag[0] == "Roemu"))
    don = ((donnerstag[0] == "Vaettu" and donnerstag[1] == "Roemu") or (donnerstag[1] == "Vaettu" and donnerstag[0] == "Roemu"))
    sam = (samstag[0] == "Vaettu" or samstag[1] == "Vaettu" or samstag[2] == "Vaettu")
    return (fri or don or sam)

while constraints():
    samstag = ["Gaebu", "Stefu", "Maettu"]
    freitag = ["Paerli", "Roemu"]
    donnerstag = ["Eric", "Vaettu"]
    teilnehmer = ["Eric", "Paerli", "Gaebu", "Maettu", "Vaettu", "Roemu", "Stefu"]
    # samstag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    # samstag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    # samstag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    samstag.append(zutaten.pop(randint(0, len(zutaten)-1)))
    samstag.append(alk.pop(randint(0, len(alk)-1)))
    # freitag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    # freitag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    freitag.append(zutaten.pop(randint(0, len(zutaten)-1)))
    freitag.append(alk.pop(randint(0, len(alk)-1)))
    # donnerstag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    # donnerstag.append(teilnehmer.pop(randint(0, len(teilnehmer)-1)))
    donnerstag.append(zutaten.pop(randint(0, len(zutaten)-1)))
    donnerstag.append(alk.pop(randint(0, len(alk)-1)))


print("\nDonnsti: ")
for i in donnerstag:
    print(i)

print("\nFriti: ")
for i in freitag:
    print(i)

print("\nSamsti: ")
for i in samstag:
    print(i)