import streamlit as st
from graph_utils import Graph, open_graph_txt, MinimumPath, Components
from graph_utils import DFSL as DFS
from graph_utils import BFSL as BFS

import mpu
import base64
import json
import numpy as np
import pandas as pd
import os

def get_table_download_link_csv(df, filename="file.txt", label="Download file", index=False):
    csv = df.to_csv(index=index).encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" target="_blank">{label}</a>'
    return href

def main():
    st.title("COS242 - Primeiro trabalho prático")
    st.header("UFRJ - Escola Politécnica - Eng. de Computação e Informação")
    menu = ["Carregar grafo", "Representação do grafo", "Estatísticas", 
    "BFS", "DFS", "Caminhos mínimos", "Componentes conexas"]
    choice = st.sidebar.selectbox("Opções", menu)
    
    graph_txt = None
    graph = None

    

    if choice == "Carregar grafo":
        st.subheader("Carregar um grafo através de um arquivo txt")
        st.markdown("""
        <div>
        <p>Faça upload de um arquivo contendo um grafo, onde a primeira linha é o número de vértices, as próximas linhas são as arestas, e a última linha em branco ou com um ponto.</p>
        <p><b>Exemplo:</b></p>
        <p>3<br>1 2<br>2 3<br>.</p>
        </div>
        """, unsafe_allow_html = True)
        graph_txt = st.file_uploader("Upload do Grafo (.txt)", type="txt")
        if graph_txt:
            graph_txt_button = st.button("Ler Grafo")
            if graph_txt_button:
                graph = open_graph_txt(graph_txt)
                graph.sort_neighbors()
                mpu.io.write("graph.pickle", graph)
                st.success(f"Grafo com {graph.n_nodes} nós carregado com sucesso!")
        st.header("Limpar grafo")
        limpar = st.button("Limpar")
        if limpar:
            try:
                os.remove("graph.pickle")
            except:
                pass
            st.success("Grafo excluído com sucesso!")
    
    elif choice == "Representação do grafo":
        st.subheader("Representação do grafo")
        st.markdown("""
        <div>
        <p>Um grafo pode ser tanto representado como uma matriz de adjacência, como listas de adjacência.</p>
        <p>Escolha a opção desejada para baixar o arquivo de representação.</p>
        </div>
        """, unsafe_allow_html = True)
        try:
            graph = mpu.io.read("graph.pickle")
            matriz = st.button("Gerar matriz de adjacência")
            listas = st.button("Gerar listas de adjacência")
            if matriz:
                st.success("Matriz de adjacência gerada com sucesso!")
                st.markdown(get_table_download_link_csv(graph.get_matrix_beautiful(), "matriz.csv", "Download matriz de adjacência (.csv)", index=True), 
                unsafe_allow_html=True)
            if listas:
                lista_json = json.dumps(graph.get_node_edges()).encode()
                b64 = base64.b64encode(lista_json).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="listas.txt" target="_blank">Download listas de adjacência (.txt)</a>'
                st.success("Listas de adjacência gerada com sucesso!")
                st.markdown(href, unsafe_allow_html=True)
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")
    
    elif choice == "Estatísticas":
        st.header("Estatísticas do grafo")
        try:
            graph = mpu.io.read("graph.pickle")
            st.markdown(f"""
            <div>
            <p>Número de vértices: {int(graph.n_nodes)}</p>
            <p>Número de arestas: {int(graph.get_matrix().sum()/2)}</p>
            <p>Grau mínimo: {int(graph.get_matrix().sum(axis=0).min())}</p>
            <p>Grau máximo: {int(graph.get_matrix().sum(axis=0).max())}</p>
            <p>Grau médio: {graph.get_matrix().sum(axis=0).mean()}</p>
            <p>Mediana do Grau: {np.median(graph.get_matrix().sum(axis=0))}</p>
            </div>
            """, unsafe_allow_html = True)

        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")

    elif choice == "BFS":
        st.header("Busca em lagura (BFS)")
        try:
            graph = mpu.io.read("graph.pickle")
            st.text(f"Escolha qual nó para ser a raiz (1 - {graph.n_nodes}) e pressione Enter para atualizar o valor")
            root = st.text_input("Raiz", 1)
            if root:
                button = st.button("Rodar BFS")
                if button:
                    try:
                        bfs = BFS(graph, int(root))
                        bfs.search()
                        st.success(f"BFS rodada com sucesso!")
                        st.markdown(get_table_download_link_csv(pd.DataFrame(list(zip(range(1, bfs.graph.n_nodes+1), bfs.level, bfs.parent)), columns=["node", "level", "parent"], index=np.arange(1, bfs.graph.n_nodes+1)),
                            f"bfs_{root}.csv", f"Download BFS do vértice {root} (.csv)"), unsafe_allow_html=True)
                    except:
                        st.error("Raiz informada não contém no grafo.")
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")

    elif choice == "DFS":
        st.header("Busca em profundidade (DFS)")
        try:
            graph = mpu.io.read("graph.pickle")
            st.text(f"Escolha qual nó para ser a raiz (1 - {graph.n_nodes}) e pressione Enter para atualizar o valor")
            root = st.text_input("Raiz", 1)
            if root:
                button = st.button("Rodar DFS")
                print(root)
                if button:
                    try:
                        dfs = DFS(graph, int(root))
                        dfs.search()
                        st.success(f"DFS rodada com sucesso!")
                        st.markdown(get_table_download_link_csv(pd.DataFrame(list(zip(range(1, dfs.graph.n_nodes+1), dfs.level, dfs.parent)), columns=["node", "level", "parent"], index=np.arange(1, dfs.graph.n_nodes+1)),
                            f"dfs_{root}.csv", f"Download DFS do vértice {root} (.csv)"), unsafe_allow_html=True)
                    except:
                        st.error("Raiz informada não contém no grafo.")
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")
    
    elif choice == "Caminhos mínimos":
        st.header("Calcular distância entre dois nós")
        try:
            graph = mpu.io.read("graph.pickle")
            minpath = MinimumPath(graph)         
            v = st.text_input("Origem", 1)
            w = st.text_input("Destino", 2)
            button = st.button("Calcular")
            if button and minpath:
                st.text(f"Diâmetro do grafo: {minpath.get_diameter()}")
                st.text(f"Distância entre {v} e {w}: {minpath.get_distance(int(v), int(w))}")
                st.header("Distância entre todos os vértices")
                st.markdown(get_table_download_link_csv(minpath.get_matrix_beautiful(), "distancias.csv", "Download matriz com todas as distâncias (.csv)", index=True), unsafe_allow_html=True)
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")

    elif choice == "Componentes conexas":
        st.header("Componentes conexas")
        try:
            graph = mpu.io.read("graph.pickle")
            button = st.button("Gerar componentes conexas")
            if button:
                components = Components(graph)
                components_dict = {len(x):x for x in components.components}
                components_json = json.dumps(components_dict).encode()
                b64 = base64.b64encode(components_json).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="componentes.txt" target="_blank">Download das componentes (.txt) </a>'
                st.success(f"Componentes geradas com sucesso!")
                st.markdown(href, unsafe_allow_html=True)
        except:
            st.error("Você ainda não carregou o grafo. Escolha a opção 'Carregar grafo' no menu e carregue seu grafo.")
    
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.text("Criado por Henrique Chaves e Pedro Boechat - 2020.2")

    if not graph:
        st.sidebar.text("Nenhum grafo carregado.")
    
    else:
        st.sidebar.text(f"Grafo com {graph.n_nodes} vértices carregado.")



if __name__ == "__main__":
    main()





