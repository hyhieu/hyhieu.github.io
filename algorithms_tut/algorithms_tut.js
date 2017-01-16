function Graph(num_nodes) {
  this.n = num_nodes;
  this.m = 0;
  this.adj = [];
  for (var i = 0; i < this.n; ++i) {
    this.adj.push([]);
  }

  this.AddNode = function() {
    (this.n)++;
    this.adj.push([]);
  };

  this.AddEdge = function(u,v) {
    (this.m)++;
    adj[u].push(v);
  };
};

var G = new Graph(0);
console.log('Created a graph G with ' + G.n + ' nodes');

DFS = {
  main_frame: document.getElementById('dfs_main'),

  MouseIn: function() {
    this.main_frame.innerHTML = 'World';
  },
  
  MouseOut: function() {
    this.main_frame.innerHTML = 'Hello';
  }
};
