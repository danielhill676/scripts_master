import random
import csv
from datetime import datetime, timedelta


Cosmo = ['Adi', 'Elliot', 'Selin', 'Niek', 'Rintaro', 'Karthika', 'Carolyn']
Stellar = ['Logan', 'Betsy', 'Lucas', 'Federica', 'Peterjan', 'Ankur']
AGN = ['Dan', 'Rose', 'Rion', 'Paloma']
Compact = ['Charlie', 'Nico', 'Andrei']


def shuffle_lists(lists, n_pick, start_list, max_attempts=3):

    for i, (lst, n) in enumerate(zip(lists, n_pick)):
        if len(lst) < n:
            raise ValueError(
                f"List {i + 1} only contains {len(lst)} elements, "
                f"but {n} were requested."
            )

    selected = [
        random.sample(lst, n)
        for lst, n in zip(lists, n_pick)
    ]

    best_result = None

    for attempt in range(max_attempts):

        remaining = [lst.copy() for lst in selected]

        result = []
        last_list = None
        current_list = start_list

        valid = True

        while any(remaining):

            # If the current list has run out of elements,
            # choose another available list
            if not remaining[current_list]:

                available = [
                    i for i in range(len(remaining))
                    if remaining[i] and i != last_list
                ]

                if not available:
                    valid = False
                    break

                current_list = random.choice(available)

            element = random.choice(remaining[current_list])
            remaining[current_list].remove(element)

            result.append(element)
            last_list = current_list

            # Choose the next list, excluding the one just used
            available = [
                i for i in range(len(remaining))
                if remaining[i] and i != last_list
            ]

            if not available:
                if any(remaining):
                    valid = False
                break

            current_list = random.choice(available)

        best_result = result

        # If successful, return immediately
        if valid and len(result) == sum(n_pick):
            print(f"Valid shuffle found on attempt {attempt + 1}.")
            return result

    # No valid solution found within max_attempts
    print(
        f"No valid shuffle found after {max_attempts} attempts. "
        "Returning the final attempt."
    )

    return best_result


# ---------------------------------------------------------
# Lists
# ---------------------------------------------------------

list1 = Cosmo
list2 = Stellar
list3 = AGN
list4 = Compact

lists = [list1, list2, list3, list4]

# ---------------------------------------------------------
# Number to pick from each list
# ---------------------------------------------------------

n_pick = [6, 6, 4, 3]
start_list = 1 # Start with stellar

# ---------------------------------------------------------
# Shuffle
# ---------------------------------------------------------

shuffled = shuffle_lists(
    lists,
    n_pick,
    start_list=start_list,
    max_attempts=3
)


print("\nFinal list:")
print(shuffled)

print("\nReserve:")
for i, (lst, selected_n) in enumerate(zip(lists, n_pick)):
    not_picked = [element for element in lst if element not in shuffled]
    print(f"List {i + 1}: {not_picked}")


# ---------------------------------------------------------
# Write to CSV
# ---------------------------------------------------------

output_file = "/Users/administrator/Documents/JC_arxiv_schedule_2026to7.csv"

start_date = datetime.strptime("08/09/2026", "%d/%m/%Y")

rows = []

current_date = start_date
name_index = 0
row_number = 0

while name_index < len(shuffled):

    if current_date >= datetime.strptime("15/12/2026", "%d/%m/%Y") \
            and current_date < datetime.strptime("12/01/2027", "%d/%m/%Y"):

        current_date = datetime.strptime("12/01/2027", "%d/%m/%Y")

    if row_number % 2 == 1:
        name = shuffled[name_index]
        name_index += 1
    else:
        name = ""

    rows.append([
        current_date.strftime("%d/%m/%Y"),
        name+' (arxiv)'
    ])

    current_date += timedelta(days=7)
    row_number += 1


# Write CSV
with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Date", "Name"])
    writer.writerows(rows)

print(f"\nCSV written to: {output_file}")