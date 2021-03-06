def find_end_of_subtree(sentence, pos):
  if sentence[pos] != "(":
    return pos
  sum = 1
  while sum > 0:
    pos += 1
    if (pos >= len(sentence)):
      return -1
    if sentence[pos] == "(":
      sum += 1
    elif sentence[pos] == ")":
      sum -= 1
    if sum == 0:
      return pos
  return -1

def is_parsable(sentence):
  if len(sentence) == 1:
    return True

  if sentence[0] != '(':
    return False
  if find_end_of_subtree(sentence, 0) + 1 != len(sentence):
    return False
  if len(sentence) == 3:
    return True

  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    if npos == 0:
      return False
    if not is_parsable(sentence[pos:npos]):
      return False
    pos = npos

  return True

def split_into_subtrees(sentence, max_size):
  if len(sentence) <= 1:
    return []

  if sum(map(lambda x: 0 if (x == '(' or x == ')') else 1, sentence)) <= max_size:
    return [sentence]

  ret = []
  pos = 2
  while pos + 1 < len(sentence):
    npos = find_end_of_subtree(sentence, pos) + 1
    ret.extend(split_into_subtrees(sentence[pos:npos], max_size))
    pos = npos

  return ret
