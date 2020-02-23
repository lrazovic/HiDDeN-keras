def string_to_binary(string):
    # UTF-8 Encoding
    return ' '.join(format(ord(x), 'b') for x in string)
