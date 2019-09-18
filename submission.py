import numpy as np
import re

def findTOP_K(a,b,K):
    for i in range(len(a)):
        a[i] = [a[i],b[i]]
    if len(a)<= K:
        return modifyForm(a)
    a = sorted(a,key=lambda x:x[1],reverse=True)
    candidates = a[:K]
    for i in range(K,len(a)):
        if a[i][1] == a[K-1][1]:
            candidates.append(a[i])
        else:
            break
    if len(candidates)<=K:
        return modifyForm(candidates)
    for i in range(len(candidates)-1,-1,-1):
        if candidates[i][1]!=candidates[-1][1]:
            sameProbility = candidates[i+1:]
            sameProbility = sortBetterPath(sameProbility)
            candidates = candidates[:i + 1] + sameProbility[:K - i - 1]
            break
        elif i == 0 and candidates[i][1]==candidates[-1][1]:
            sameProbility = sortBetterPath(candidates)
            candidates = sameProbility[:K]
    return modifyForm(candidates)

def sortBetterPath(sameProbility):
    for j in range(len(sameProbility[0][0])):
        sameProbility = sorted(sameProbility, key=lambda x: x[0][j])
    return sameProbility

def modifyForm(candidates):
    for i in range(len(candidates)):
        candidates[i][0].append(candidates[i].pop())
        candidates[i] = candidates[i][0]
    return candidates


def init_query(Query_File):
    Query_File_data = open(Query_File, 'r')
    Query_File_data_lines = Query_File_data.read().splitlines()
    query_list = []
    punctuation_list = r'([-,()&/ ])'
    for data in Query_File_data_lines:
        l=[]
        data_list=re.split(punctuation_list, data)

        for symbol in data_list:
            symbol_without_blank=symbol.strip()
            if len(symbol_without_blank)!=0:
                l.append(symbol_without_blank)
        query_list.append(l)
    return query_list

def init_state_matrix(State_File):
    State_File_data = open(State_File, 'r')
    State_File_data_lines = State_File_data.read().splitlines()
    State_number = int(State_File_data_lines[0])
    State_number_except_begin_end=State_number-2
    State_matrix=np.zeros((State_number,State_number))
    State_data = State_File_data_lines[State_number+1:]
    for raw_data in State_data:
        f1_str, f2_str, f3_str = raw_data.split(' ')
        f1 = int(f1_str)
        f2 = int(f2_str)
        f3 = int(f3_str)
        State_matrix[f1][f2] = f3
    for i in range(len(State_matrix)):
        data_in_line=State_matrix[i]
        State_matrix[i]=(data_in_line + 1) / (sum(data_in_line) + State_number - 1)
    initial_transition_pro_list=State_matrix[State_number-2,:-2]
    return State_matrix,initial_transition_pro_list,State_number_except_begin_end




def init_symbol_matrix(Symbol_File,State_File):

    temp = open(State_File, 'r')
    temp_lines=temp.read().splitlines()
    temp_row = int(temp_lines[0])-2
    symbol_list=[]
    Symbol_File_data = open(Symbol_File, 'r')
    Symbol_File_data_lines=Symbol_File_data.read().splitlines()
    Symbol_number=int(Symbol_File_data_lines[0])
    for line_num in range(1,Symbol_number+1):
        symbol_list.append(Symbol_File_data_lines[line_num])

    Symbol_matrix=np.zeros((temp_row,Symbol_number))
    data_begin_line=Symbol_number+1
    Symbol_data=Symbol_File_data_lines[data_begin_line:]
    for raw_data in Symbol_data:
        f1_str,f2_str,f3_str=raw_data.split(' ')
        f1=int(f1_str)
        f2=int(f2_str)
        f3=int(f3_str)
        Symbol_matrix[f1][f2]=f3
    nj=Symbol_matrix.sum(axis=1)
    for i in range(len(Symbol_matrix)):
        data_in_line=Symbol_matrix[i]
        Symbol_matrix[i]=(data_in_line+1)/(sum(data_in_line)+Symbol_number+1)
    return Symbol_matrix.T,symbol_list,nj


def init_symbol_matrix_for_q3(Symbol_File,State_File):

    temp = open(State_File, 'r')
    temp_lines=temp.read().splitlines()
    temp_row = int(temp_lines[0])-2
    symbol_list=[]
    Symbol_File_data = open(Symbol_File, 'r')
    Symbol_File_data_lines=Symbol_File_data.read().splitlines()
    Symbol_number=int(Symbol_File_data_lines[0])
    for line_num in range(1,Symbol_number+1):
        symbol_list.append(Symbol_File_data_lines[line_num])

    Symbol_matrix=np.zeros((temp_row,Symbol_number+1))
    data_begin_line=Symbol_number+1
    Symbol_data=Symbol_File_data_lines[data_begin_line:]
    for raw_data in Symbol_data:
        f1_str,f2_str,f3_str=raw_data.split(' ')
        f1=int(f1_str)
        f2=int(f2_str)
        f3=int(f3_str)
        Symbol_matrix[f1][f2]=f3
    nj=Symbol_matrix.sum(axis=1)
    for i in range(len(Symbol_matrix)):
        data_in_line=Symbol_matrix[i]
        count=0
        for data in data_in_line:
            if int(data)!=0:
                count+=1
        p=1/(sum(data_in_line)+count)
        Symbol_matrix[i]=data_in_line*(1/np.sum(data_in_line))

        for idx in range(Symbol_matrix.shape[1]):
            if Symbol_matrix[i][idx]==0:
                Symbol_matrix[i][idx]=(count*p)/(Symbol_number+1-count)
            else:
                Symbol_matrix[i][idx]=Symbol_matrix[i][idx]-p
    return Symbol_matrix.T,symbol_list,nj


def viterbi_algorithm(State_File,Symbol_File,Query_File):
    result_list = []
    Symbol_matrix, symbol_list, nj = init_symbol_matrix(Symbol_File, State_File)
    Query_list = init_query(Query_File)
    relational_table = dict()
    for symbol_index in range(len(symbol_list)):
        relational_table[symbol_list[symbol_index]] = symbol_index
    for query in Query_list:
        columns = len(query)
        path_list = []
        State_matrix, initial_transition_pro_list, nb_of_state = init_state_matrix(State_File)

        end_matrix = State_matrix[:-2, -1]
        State_matrix = (State_matrix[:-2, :-2].T)
        viterbi_matrix = np.zeros((columns, nb_of_state))
        begin_state = None
        for s in range(nb_of_state):
            path_list.append([nb_of_state, s])
        M = len(symbol_list)
        top_k_path = []
        for i in range(0, columns):

            l = []
            symbol = query[i]
            temp_list = []
            if i == 0:
                if symbol not in symbol_list:
                    initial_row = 1 / (M + 1 + nj)
                    begin_state = initial_transition_pro_list * initial_row
                else:
                    row_number = relational_table[symbol]
                    begin_state = initial_transition_pro_list * Symbol_matrix[row_number]
                viterbi_matrix = np.insert(viterbi_matrix, 0, values=begin_state, axis=0)
                continue
            for j in range(nb_of_state):

                if symbol not in symbol_list:
                    emission_pro = 1 / (M + 1 + nj[j])
                else:
                    row_number = relational_table[symbol]
                    emission_pro = Symbol_matrix[row_number][j]

                if len(top_k_path) != 0:
                    transition_pro_list = []
                    for pre_state in top_k_path:
                        state = pre_state[-1]
                        transition_pro = State_matrix[j][state]
                        transition_pro_list.append(transition_pro)
                    value = np.array(begin_state) * np.array(transition_pro_list) * emission_pro
                    l.append(value)
                else:
                    transition_pro = State_matrix[j]
                    value = begin_state * transition_pro * emission_pro
                    l.append(value)
            temp_matrix = np.array(l)
            state_list = []
            for row_index in range(len(temp_matrix)):
                row_matrix = temp_matrix[row_index]
                sorted_row_list = sorted(list(row_matrix), reverse=True)
                if len(sorted_row_list) <= 1:
                    state_list += (sorted_row_list)
                else:
                    count = 1
                    for val_index in range(1, len(sorted_row_list)):
                        if sorted_row_list[1 - 1] == sorted_row_list[val_index]:
                            count += 1
                        else:
                            break
                    state_list += (sorted_row_list[:count])

            top_k_list = state_list
            top_k_path = []
            top_k_values = []
            for v in set(top_k_list):
                ordination = (np.argwhere(temp_matrix == np.float(v)))
                for index in range(len(ordination)):
                    x, y = list(ordination[index])
                    temp_path = [i for i in (path_list[y])]
                    temp_path.append(x)
                    top_k_path.append(temp_path)
                    top_k_values.append(temp_matrix[x][y])
            begin_state = top_k_values

            path_list = [i for i in top_k_path]
        end_list = []
        for last_state in top_k_path:
            end_state = last_state[-1]
            end_list.append(end_matrix[end_state])

        end_value = begin_state * np.array(end_list)
        log_values = (np.log(end_value))
        for i in range(len(top_k_path)):
            top_k_path[i].append(nb_of_state + 1)
        for i in findTOP_K(top_k_path, log_values, 1):
            result_list.append(i)

    return result_list


def top_k_viterbi(State_File, Symbol_File, Query_File,k):
    result_list=[]
    Symbol_matrix, symbol_list, nj = init_symbol_matrix(Symbol_File, State_File)
    Query_list=init_query(Query_File)
    relational_table = dict()
    for symbol_index in range(len(symbol_list)):
        relational_table[symbol_list[symbol_index]] = symbol_index
    for query in Query_list:
        columns = len(query)
        path_list = []
        State_matrix, initial_transition_pro_list, nb_of_state = init_state_matrix(State_File)

        end_matrix = State_matrix[:-2, -1]
        State_matrix = (State_matrix[:-2, :-2].T)
        viterbi_matrix = np.zeros((columns, nb_of_state))
        begin_state = None
        for s in range(nb_of_state):
            path_list.append([nb_of_state, s])
        M = len(symbol_list)
        top_k_path = []
        for i in range(0, columns):

            l = []
            symbol = query[i]
            temp_list = []
            if i == 0:
                if symbol not in symbol_list:
                    initial_row = 1 / (M + 1 + nj)
                    begin_state = initial_transition_pro_list * initial_row
                else:
                    row_number = relational_table[symbol]
                    begin_state = initial_transition_pro_list * Symbol_matrix[row_number]
                viterbi_matrix = np.insert(viterbi_matrix, 0, values=begin_state, axis=0)
                continue
            for j in range(nb_of_state):

                if symbol not in symbol_list:
                    emission_pro = 1 / (M + 1 + nj[j])
                else:
                    row_number = relational_table[symbol]
                    emission_pro = Symbol_matrix[row_number][j]

                if len(top_k_path)!=0:
                    transition_pro_list=[]
                    for pre_state in top_k_path:
                        state=pre_state[-1]
                        transition_pro = State_matrix[j][state]
                        transition_pro_list.append(transition_pro)
                    value = np.array(begin_state) * np.array(transition_pro_list) * emission_pro
                    l.append(value)
                else:
                    transition_pro = State_matrix[j]
                    value = begin_state * transition_pro * emission_pro
                    l.append(value)
            temp_matrix=np.array(l)
            state_list=[]
            for row_index in range(len(temp_matrix)):
                row_matrix=temp_matrix[row_index]
                sorted_row_list=sorted(list(row_matrix),reverse=True)
                if len(sorted_row_list)<=k:
                    state_list+=(sorted_row_list)
                else:
                    count=k
                    for val_index in range(k,len(sorted_row_list)):
                        if sorted_row_list[k-1]==sorted_row_list[val_index]:
                            count+=1
                        else:
                            break
                    state_list+=(sorted_row_list[:count])

            top_k_list=state_list
            top_k_path=[]
            top_k_values=[]
            for v in set(top_k_list):
                ordination=(np.argwhere(temp_matrix == np.float(v)))
                for index in range(len(ordination)):
                    x,y=list(ordination[index])
                    temp_path=[i for i in (path_list[y])]
                    temp_path.append(x)
                    top_k_path.append(temp_path)
                    top_k_values.append(temp_matrix[x][y])
            begin_state=top_k_values

            path_list=[i for i in top_k_path]
        end_list=[]
        for last_state in top_k_path:
            end_state=last_state[-1]
            end_list.append(end_matrix[end_state])

        end_value = begin_state * np.array(end_list)
        log_values=(np.log(end_value))
        for i in range(len(top_k_path)):
            top_k_path[i].append(nb_of_state+1)
        for i in findTOP_K(top_k_path, log_values,k):
            result_list.append(i)

    return result_list



def advanced_decoding(State_File, Symbol_File, Query_File):
    result_list = []
    Symbol_matrix, symbol_list, nj = init_symbol_matrix_for_q3(Symbol_File, State_File)
    Query_list = init_query(Query_File)
    relational_table = dict()
    for symbol_index in range(len(symbol_list)):
        relational_table[symbol_list[symbol_index]] = symbol_index
    for query in Query_list:
        columns = len(query)
        path_list = []
        State_matrix, initial_transition_pro_list, nb_of_state = init_state_matrix(State_File)

        end_matrix = State_matrix[:-2, -1]
        State_matrix = (State_matrix[:-2, :-2].T)
        viterbi_matrix = np.zeros((columns, nb_of_state))
        begin_state = None
        for s in range(nb_of_state):
            path_list.append([nb_of_state, s])
        M = len(symbol_list)
        top_k_path = []
        for i in range(0, columns):

            l = []
            symbol = query[i]
            temp_list = []
            if i == 0:
                if symbol not in symbol_list:
                    initial_row = Symbol_matrix[-1, :]
                    begin_state = initial_transition_pro_list * initial_row
                else:
                    row_number = relational_table[symbol]
                    begin_state = initial_transition_pro_list * Symbol_matrix[row_number]
                viterbi_matrix = np.insert(viterbi_matrix, 0, values=begin_state, axis=0)
                continue
            for j in range(nb_of_state):

                if symbol not in symbol_list:
                    emission_pro = Symbol_matrix[-1, :][j]
                else:
                    row_number = relational_table[symbol]
                    emission_pro = Symbol_matrix[row_number][j]

                if len(top_k_path) != 0:
                    transition_pro_list = []
                    for pre_state in top_k_path:
                        state = pre_state[-1]
                        transition_pro = State_matrix[j][state]
                        transition_pro_list.append(transition_pro)
                    value = np.array(begin_state) * np.array(transition_pro_list) * emission_pro
                    l.append(value)
                else:
                    transition_pro = State_matrix[j]
                    value = begin_state * transition_pro * emission_pro
                    l.append(value)
            temp_matrix = np.array(l)
            state_list = []
            for row_index in range(len(temp_matrix)):
                row_matrix = temp_matrix[row_index]
                sorted_row_list = sorted(list(row_matrix), reverse=True)
                if len(sorted_row_list) <= 1:
                    state_list += (sorted_row_list)
                else:
                    count = 1
                    for val_index in range(1, len(sorted_row_list)):
                        if sorted_row_list[1 - 1] == sorted_row_list[val_index]:
                            count += 1
                        else:
                            break
                    state_list += (sorted_row_list[:count])

            top_k_list = state_list
            top_k_path = []
            top_k_values = []
            for v in set(top_k_list):
                ordination = (np.argwhere(temp_matrix == np.float(v)))
                for index in range(len(ordination)):
                    x, y = list(ordination[index])
                    temp_path = [i for i in (path_list[y])]
                    temp_path.append(x)
                    top_k_path.append(temp_path)
                    top_k_values.append(temp_matrix[x][y])
            begin_state = top_k_values

            path_list = [i for i in top_k_path]
        end_list = []
        for last_state in top_k_path:
            end_state = last_state[-1]
            end_list.append(end_matrix[end_state])

        end_value = begin_state * np.array(end_list)
        log_values = (np.log(end_value))
        for i in range(len(top_k_path)):
            top_k_path[i].append(nb_of_state + 1)
        for i in findTOP_K(top_k_path, log_values, 1):
            result_list.append(i)

    return result_list


