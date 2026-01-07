import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, adjacency, labels=False, positions=None,zone_lookup_df=None):
        """
        Initialize a graph from an adjacency matrix (numpy array).
        Class member variables:
        A: Adjacency
        n: number of nodes
        labels: the labels of the nodes (e.g. indeces)
        zone_lookup_df:
        G: networkx graph
        D: Degree matrix
        L: Laplacian
        L_sym: normalized Laplacian
        eigvals,eigvecs: eigenvalues/-vectors based on the normalized laplacian

        Parameters
        ----------
        adjacency : np.ndarray
            Adjacency matrix (NxN)
        labels : bool
            Whether to display node labels
        positions : dict
            Node positions (optional). 
            - If dict: {node_index: (x, y), ...}
        zone_lookup_df: pandas.df
            a table matching the index to the borough name
            
        """
        self.A = np.array(adjacency, dtype=float)
        self.n = self.A.shape[0]
        self.labels = labels
        self.zone_lookup_df=zone_lookup_df

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Adjacency matrix must be square.")

        self.G = nx.from_numpy_array(self.A)

        # --- store positions ---
        if positions is not None:
            if isinstance(positions, np.ndarray):
                if positions.shape != (self.n, 2):
                    raise ValueError("positions array must have shape (N, 2)")
                self.positions = {i: tuple(pos) for i, pos in enumerate(positions)}
            elif isinstance(positions, dict):
                self.positions = positions
            else:
                raise TypeError("positions must be a dict or a numpy array of shape (N, 2)")
        else:
            self.positions = None

        # Degree and Laplacian matrices
        self.D = np.diag(self.A.sum(axis=1))
        self.L = self.D - self.A
        self.L_sym = self._normalized_laplacian()
        self.eigvalsL,self.eigvecsL = self.spectrum(matrix_type="normalized_laplacian")

    # ---------------------- CORE ----------------------

    def _normalized_laplacian(self):
        d = np.diag(self.D)
        inv_sqrt = np.zeros_like(d, dtype=float)
        mask = d > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(inv_sqrt)
        return np.eye(self.n) - D_inv_sqrt @ self.A @ D_inv_sqrt

    # ---------------------- SPECTRAL ----------------------
    def measure_of_variation(self,signal):
        # Calculates measure of variation as proposed in "Graph Signal Processing - Overview, Challenges and Applications"
        # Only needed for graphs where the adjacency is the shift operator
        vals, _ = self.spectrum(matrix_type="adjacency")
        A_norm = self.A / np.max(vals)
        return np.sum(np.abs(signal - A_norm @ signal))


    def spectrum(self, matrix_type):
        if matrix_type == "adjacency":
            M = self.A
        elif matrix_type == "laplacian":
            M = self.L
        elif matrix_type == "normalized_laplacian":
            M = self.L_sym
        else:
            raise ValueError("matrix_type must be adjacency, laplacian, or normalized_laplacian")

        if np.allclose(M, M.T):
            vals, vecs = np.linalg.eigh(M)
        else:
            vals, vecs = np.linalg.eig(M)

        if matrix_type != "adjacency":
            idx = np.argsort(vals)
            vals, vecs = vals[idx], vecs[:, idx]
            vals[0] = 0
        else:
            lambda_max = np.max(np.abs(vals))
            A_norm = self.A / lambda_max
            variation = np.array([np.sum(np.abs(vecs[:, i] - A_norm @ vecs[:, i])) for i in range(self.n)])
            idx = np.argsort(variation)
            vals, vecs = vals[idx], vecs[:, idx]
        return vals, vecs

    def gft(self, x, matrix_type="laplacian"):
        vals, vecs = self.spectrum(matrix_type)
        return vecs.conj().T @ x, vals, vecs
    
    def fast_gft(self,x):
        return self.eigvecsL.conj().T @ x

    def igft(self, x_hat, vecs):
        return vecs @ x_hat

    # ---------------------- PLOTTING ----------------------

    def plot(self, node_values=None, colormap='plasma', node_size=100, vmin=None, vmax=None,
             save_plot=False, plot_name=None, show_title=True):
        """
        Plot the graph with optional fixed node positions.
        """
        
        pos = self.positions if self.positions is not None else nx.spring_layout(self.G, seed=42)

        edges, weights = zip(*nx.get_edge_attributes(self.G, 'weight').items())
        weights = np.array(weights, dtype=float)
        w_min, w_max = weights.min(), weights.max()
        widths = 0.5 + 5.5 * (weights - w_min) / (w_max - w_min + 1e-9)
        fig, ax = plt.subplots(figsize=(6, 6))
        if node_values is None:
            nodes = nx.draw_networkx_nodes(
                self.G, pos,
                node_size=node_size
            )
        else:
            if vmin is None:
                vmin = node_values.min()
            if vmax is None:
                vmax = node_values.max()
            nodes = nx.draw_networkx_nodes(
                self.G, pos,
                node_color=node_values,
                node_size=node_size,
                cmap=plt.get_cmap(colormap),
                vmin=vmin,
                vmax=vmax
            )

        
        
        nx.draw_networkx_edges(self.G, pos, edgelist=edges, width=widths, edge_color='gray', alpha=0.8)
        if self.labels:
            nx.draw_networkx_labels(self.G, pos)
        cbar = plt.colorbar(nodes, label='Node values')
        cbar.ax.tick_params(labelsize=21)
        cbar.set_label('Node values', fontsize=17)
        plt.axis('off')
        
        if plot_name is not None and show_title is True:
            plt.title(plot_name)

        if save_plot:
            if plot_name is None:
                plot_name = "graph_plot"
            plt.savefig(f"../Plots/{plot_name}.pgf", bbox_inches='tight', dpi=300)
            print(f"Plot saved as {plot_name}.pgf")

        plt.show()
        plt.close(fig)
    
    def plot_spectrum(self, matrix_type, colormap='plasma', max_four=False, save_plot=False, plot_name=None):
        # Plots the eigenvectors of the specified matrix type
        if matrix_type is None:
            raise ValueError("Please indicate the matrix_type (laplacian, normalized_laplacian or adjacency).")
        vals, vecs = self.spectrum(matrix_type)
        num_plots = 4 if max_four else self.n 
        num_plots = min(num_plots, self.n)

        fig, axes = plt.subplots(nrows=num_plots, figsize=(8, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(vecs[:, i], marker='o')
            ax.set_title(f"Eigenvector {i + 1} of the {matrix_type.replace('_', ' ').title()}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")

        plt.tight_layout()
        if save_plot:
            if plot_name is None:
                plot_name = f"spectrum_{matrix_type}"
            plt.savefig(f"{plot_name}.pgf", bbox_inches='tight', dpi=300)
            print(f"Spectrum plot saved as {plot_name}.pgf")
        plt.show()
        plt.close(fig)
    
    def plot_eigvecs(self, matrix_type, eigvec_list=None, colormap="plasma", savefig=False, title=None, show_title=True, node_size=400):
        if matrix_type is None:
            raise ValueError("Please indicate the matrix_type (laplacian, normalized_laplacian or adjacency).")
        if eigvec_list is None:
            eigvec_list = [i for i in range(0, self.n)]
        eigvals, eigvecs = self.spectrum(matrix_type)
        for eigvec_index in eigvec_list:
            if eigvec_index >= self.n or eigvec_index < 0:
                raise ValueError("Eigvec index list must only contain valid indeces for eigenvectors (0 to n-1)!")
            if title is not None:
                plot_name = title + f" - Eigenvalue: {eigvals[eigvec_index]:.02f}, Eigenvector: {eigvec_index}"
            else:
                plot_name = f"Eigenvalue: {eigvals[eigvec_index]:.02f}, Eigenvector: {eigvec_index}"
            self.plot(node_values = eigvecs[:, eigvec_index], vmin=np.max(eigvecs), vmax=np.min(eigvecs), save_plot=savefig, plot_name=plot_name, show_title=show_title, node_size=node_size)
            print(f"Eigenvector index: {eigvec_index}, Eigenvalue: {eigvals[eigvec_index]}")
            if matrix_type == "adjacency":
                print(f"Measure of variation: {self.measure_of_variation(eigvecs[:, eigvec_index])}")
    
    def plot_spectral_energy(self, signal, matrix_type="normalized_laplacian", savefig = False, title = None, lf_components = 3):
        x_hat, eigvals, eigvecs = self.gft(signal, matrix_type)
        plt.figure()
        plt.plot(eigvals, 100*np.abs(x_hat)**2/np.linalg.norm(x_hat)**2, '.')
        plt.xlabel("$\lambda$ (graph frequency)")
        plt.ylabel("Percentage of the energy of a mode in $\%$")
        # plt.title("GFT spectrum of given signal")
        plt.yticks([0,20,40,60,80,100])
        if savefig:
            if title is not None:
                plt.savefig(f"../Plots/{title}.pgf", bbox_inches='tight', dpi=300)
            else:
                plt.savefig("/Plots/spectral_energy.pgf", bbox_inches='tight', dpi=300)
        # Sort coefficients from largest to smallest
        print(f"Spectral energy of first {lf_components} modes: {(x_hat[0:lf_components]**2).sum() / (((x_hat)**2).sum())}")
                
    def gwt(self,x,s,matrix_type):
        """_summary_

        Args:
            x (np.array): signal to analyze
            s (np.array): scaling factor sampling points
            matrix_type (_type_): _description_

        Returns:
            np.array: wavelet transforms at the given scaling factor samling points using the kernel from the paper
        """
        vals, vecs = self.spectrum(matrix_type)
        results=np.zeros(shape=(vals.shape[0],s.shape[0]))
        for i,s_ in enumerate(s):
            g_simple=lambda x:np.pow(x,2) if x<1 else ((-5+11*x-6*np.pow(x,2)+np.pow(x,3)) if x<2 else 4*np.pow(x,-2))
            g=np.vectorize(g_simple)
            gs=g(s_*vals)
            #plot can be used to show the kernels
            #plt.plot(vals*s_,gs,label=i)
            results[:,i]=vecs@np.diag(gs)@vecs.conj().T@x
        #plt.legend=True
        #plt.show()
        return results
                
    def plot_wavelet_spectrum(self,node_values,nodesize = 100,ns=10,colormap="plasma", matrix_type="laplacian",
                save_plot=False, plot_name=None,plot_abs=False):
        """
        Plot the Wavelet transform sampled at ns Ses in a range representative of the spectrum
            Parameters
        ----------
        signal  : np.array
            graph signal to be analysed
        ns  :  int
            number of "frequencies to probe at"
        colormap : str
            Matplotlib colormap for node coloring.
        matrix_type : str
            Type of matrix to use for the spectrum: 'adjacency', 'laplacian', or 'normalized_laplacian'.
        save_plot : bool
            If True, save each eigenvector plot as a PNG file.
        plot_name : str or None
            Base name for saved plots. If None, defaults to "graph_eig".
        """
        #gft used for GWT and to set limits for s
        ft,vals,vecs=self.gft(node_values,matrix_type)
        #set according to "Wavelets on Grophs via Spectral Graph Theory" page 26
        
        for i in vals:
            if i>1e-12:
                t1=i
                break
        t1=np.log(2/t1)
        tend=np.log(2/vals[-1])
        #number of Ses -> probe at n "frequencies"
        #array of Ses
        s=np.linspace(t1,tend,ns)
        s=np.exp(s)
        results=self.gwt(node_values,s,matrix_type)

            
    #            results=self.gwt(np.abs(signal-np.sum(signal)/np.sum(self.node_values)*self.node_values),s,matrix_type)
            
            
        for n,x in enumerate(results.T):              
            S=Graph(self.A, positions=self.positions)
            ind=np.abs(x).argsort()[-6:]
            if not self.zone_lookup_df is None:
                for i in ind:
                    city = self.zone_lookup_df.loc[i+1, ["Borough","Zone"]]
                    print(city.values,i)
            if plot_name is None:
                print(f"wavelet-spectrum-{n},{results.T[n][ind]}")
                #print(s[-(1+n)])
                x=np.abs(x) if plot_abs else x
                S.plot(colormap=colormap,node_values=x, node_size=nodesize,save_plot=save_plot, plot_name=f"wavelet-spectrum-{n}",show_title=False)
            else:
                print(f"wavelet-spectrum-{n},{results.T[n][ind]}")
                #print(s[-(1+n)])
                new_name=plot_name+f"{n}"
                S.plot(colormap=colormap,node_values=x, node_size=nodesize,save_plot=save_plot, plot_name=new_name,show_title=False)
            
