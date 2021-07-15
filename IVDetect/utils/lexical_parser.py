#from https://github.com/AkshayPatil0/lexical-analyser/blob/master/lex.py
import re


def get_sub_token(_token):
    _sub_tokens =[]
    splits = _token.split('_')
    for split in splits:
        if len(split) > 1:
            _sub_tokens.append(split)
    return _sub_tokens

def lexical_parser(text):
    symbols = ['!', '@', '#', '$', '%', '&', '^', '*']
    oparators = ['+', '-', '*', '/', '=', '+=', '-=', '==', '<', '>', '<=', '>=']
    keywords = ['auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do',
                'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if',
                'int', 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static',
                'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while']
    delimiters = [' ', '	', '.', ',', '\n', ';', '(', ')', '<', '>', '{', '}', '[', ']']

    in_keywords = []
    in_spl_symbols = []
    in_oparators = []
    in_delimiters = []
    in_identifiers = []
    in_constants = []

    tokens = []
    isStr = False
    isWord = False
    isCmt = 0
    token = ''

    for i in text:
        if i == '/':
            isCmt = isCmt + 1

        elif isCmt == 2:
            if i == '\n':
                token = ''
                isCmt = 0

        elif i == '"' or i == "'":
            if isStr:
                tokens.append(token)
                token = ''
            isStr = not isStr

        elif isStr:
            token = token + i

        elif i in symbols:
            tokens.append(i)

        elif i.isalnum() and not isWord:
            isWord = True
            token = i

        elif (i in delimiters) or (i in oparators):
            if token:
                tokens.append(token)
                token = ''

            if not (i == ' ' or i == '\n' or i == '	'):
                tokens.append(i)

        elif isWord:
            token = token + i

    for token in tokens:
        if token in symbols:
            in_spl_symbols.append(token)

        elif token in oparators:
            in_oparators.append(token)

        elif token in keywords:
            in_keywords.append(token)

        elif re.search("^[_a-zA-Z][_a-zA-Z0-9]*$", token):
             #print("token",token)
            sub_tokens = get_sub_token(token)
            for sub_token in sub_tokens:
                in_identifiers.append(sub_token)

        elif token in delimiters:
            in_delimiters.append(token)

        else:
            in_constants.append(token)

    # print("No of tokens = ", len(tokens))
    # print("\nNo. of keywords = ", len(in_keywords))
    # print(in_keywords)
    # print("\nNo. of special symbols = ", len(in_spl_symbols))
    # print(in_spl_symbols)
    # print("\nNo. of oparators = ", len(in_oparators))
    # print(in_oparators)
    # print("\nNo. of identifiers = ", len(in_identifiers))
    # print(in_identifiers)
    # print("\nNo. of constants = ", len(in_constants))
    # print(in_constants)
    # print("\nNo. of delimiters = ", len(in_delimiters))
    # print(in_delimiters)
    return in_identifiers


# code = "(<operator>.indirectFieldAccess,s->reinit)"


# code = """static void v4l2_free_buffer(void *opaque, uint8_t *unused)
#
# {
#
#     V4L2Buffer* avbuf = opaque;
#
#     V4L2m2mContext *s = buf_to_m2mctx(avbuf);
#
#
#
#     if (atomic_fetch_sub(&avbuf->context_refcount, 1) == 1) {
#
#         atomic_fetch_sub_explicit(&s->refcount, 1, memory_order_acq_rel);
#
#
#
#         if (s->reinit) {
#
#             if (!atomic_load(&s->refcount))
#
#                 sem_post(&s->refsync);
#
#         } else if (avbuf->context->streamon)
#
#             ff_v4l2_buffer_enqueue(avbuf);
#
#
#
#         av_buffer_unref(&avbuf->context_ref);
#
#     }
#
# }"""
#
# code = "v4l2_free_buffer)"
# tokens = lexical_parser(code)
# print(tokens)