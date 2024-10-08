import math

VALID_LICHESS_MONTHS=[
    "201301-moves",
    "201302-moves",
    "201303-moves",
    "201304-moves",
    "201305-moves",
    "201306-moves",
    "201307-moves",
    "201308-moves",
    "201309-moves",
    "201310-moves",
    "201311-moves",
    "201312-moves",
    "201401-moves",
    "201402-moves",
    "201403-moves",
    "201404-moves",
    "201405-moves",
    "201406-moves",
    "201407-moves",
    "201408-moves",
    "201409-moves",
    "201410-moves",
    "201411-moves",
    "201412-moves",
    "201501-moves",
    "201502-moves",
    "201503-moves",
    "201504-moves",
    "201505-moves",
    "201506-moves",
    "201507-moves",
    "201508-moves",
    "201509-moves",
    "201510-moves",
    "201511-moves",
    "201512-moves",
    "201601-moves",
    "201602-moves",
    "201603-moves",
    "201604-moves",
    "201605-moves",
    "201606-moves",
    "201607-moves",
    "201608-moves",
    "201609-moves",
    "201610-moves",
    "201611-moves",
    "201612-moves",
    "201701-moves",
    "201702-moves",
    "201703-moves",
    "201704-moves",
    "201705-moves",
    "201706-moves",
    "201707-moves",
    "201708-moves",
    "201709-moves",
    "201710-moves",
    "201711-moves",
    "201712-moves",
    "201801-moves",
    "201802-moves",
    "201803-moves",
    "201804-moves",
    "201805-moves",
    "201806-moves",
    "201807-moves",
    "201808-moves",
    "201809-moves",
    "201810-moves",
    "201811-moves",
    "201812-moves",
    "201901-moves",
    "201902-moves",
    "201903-moves",
    "201904-moves",
    "201905-moves",
    "201906-moves",
    "201907-moves",
    "201908-moves",
    "201909-moves",
    "201910-moves",
    "201911-moves",
    "201912-moves",
    "202001-moves",
    "202002-moves",
    "202003-moves",
    "202004-moves",
    "202005-moves",
    "202006-moves",
    "202007-moves",
    "202008-moves",
    "202009-moves",
    "202010-moves",
    "202011-moves",
    "202012-moves",
    "202101-moves",
    "202102-moves",
    "202103-moves",
    "202104-moves",
    "202105-moves",
    "202106-moves",
    "202107-moves",
    "202108-moves",
    "202109-moves",
    "202110-moves",
    "202111-moves",
    "202112-moves",
    "202201-moves",
    "202202-moves",
    "202203-moves",
    "202204-moves",
    "202205-moves",
    "202206-moves",
    "202207-moves",
    "202208-moves",
    "202209-moves",
    "202210-moves",
    "202211-moves",
    "202212-moves",
    "202301-moves",
    "202302-moves",
    "202303-moves",
    "202304-moves",
    "202305-moves",
    "202306-moves",
    "202307-moves",
    "202308-moves",
    "202309-moves",
    "202310-moves",
    "202311-moves",
    "202312-moves",
    "202401-moves",
    "202402-moves",
]

def encode_list(subset: list[str], superset: list[str] = VALID_LICHESS_MONTHS, alphabet: str ='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-'):
    N = len(alphabet)
    step = int(math.log(N,2))

    set_containment = ['1' if v in subset else '0' for v in superset]
    
    encoded_str = ""

    # encode using the alphabet
    for i in range(0,len(set_containment),step):
        # Reverse so we can easily decode
        binary_str = '0b'+''.join(reversed(set_containment[i:i+step]))
        alphabet_index = int(binary_str,2)
        encoded_str += alphabet[alphabet_index]

    return encoded_str

def decode_list(encoded_str: str, superset: list[str] = VALID_LICHESS_MONTHS, alphabet: str ='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_-'):
    N = len(alphabet)
    base = int(math.log(N,2))

    char_map = {k:bin(i)[2:].zfill(base)[::-1] for i,k in enumerate(list(alphabet))}

    decoded_str = ''
    for c in encoded_str:
        decoded_str += char_map[c]

    decoded_list = []
    for i, val in enumerate(list(decoded_str)):
        if val == '1':
            decoded_list.append(superset[i])
    return decoded_list
