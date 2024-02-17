def search_and_replace(file_path, search_word, replace_word):
   with open(file_path, 'r') as file:
      file_contents = file.read()

      updated_contents = file_contents.replace(search_word, replace_word)

   with open(file_path, 'w') as file:
      file.write(updated_contents)

# Example usage
file_path = '../data/NDBC - Train/44040h2008.csv'
search_word = ''
replace_word = ','
search_and_replace(file_path, search_word, replace_word)