import React, { useMemo, useCallback } from "react";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  MarkerType,
  Position,
  Handle,
  NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Badge } from "@/components/ui/badge";
import { ConceptNode } from "@/lib/api";
import { Lock, Unlock, CheckCircle2, PlayCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// Custom node component for beautiful styling
const KnowledgeNode = ({ data, selected }: NodeProps) => {
  const { node, status, score } = data as {
    node: ConceptNode;
    status: string;
    score?: number;
  };

  const statusConfig = {
    locked: { icon: Lock, className: "bg-muted/50 border-border text-muted-foreground" },
    available: { icon: Unlock, className: "bg-card border-primary/50 text-foreground shadow-sm hover:border-primary" },
    in_progress: { icon: PlayCircle, className: "bg-primary/10 border-primary text-primary shadow-[0_0_15px_rgba(168,85,247,0.3)] ring-1 ring-primary/50" },
    mastered: { icon: CheckCircle2, className: "bg-emerald-500/10 border-emerald-500/50 text-emerald-500 shadow-sm" },
  }[status] || { icon: Lock, className: "bg-muted border-border" };

  const StatusIcon = statusConfig.icon;

  return (
    <div
      className={cn(
        "relative flex flex-col min-w-[220px] p-4 rounded-xl border-2 transition-all duration-300 backdrop-blur-md",
        statusConfig.className,
        selected && "ring-2 ring-primary ring-offset-2 ring-offset-background scale-105 shadow-xl font-medium"
      )}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3 border-2 rtl" />

      <div className="flex items-start justify-between gap-3 mb-2">
        <StatusIcon className="w-5 h-5 shrink-0 mt-0.5" />
        <h3 className="font-semibold text-sm leading-tight flex-1">{node.title}</h3>
      </div>

      <div className="flex items-center gap-2 mt-2">
        <Badge variant={status === "mastered" ? "success" : "outline"} className="text-[10px] px-1.5 py-0">
          Lvl {node.difficulty}
        </Badge>
        {score !== undefined && (
          <span className="text-xs font-semibold opacity-80">
            {Math.round(score * 100)}%
          </span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} className="w-3 h-3 border-2" />
    </div>
  );
};

const nodeTypes = {
  knowledge: KnowledgeNode,
};

interface GraphVisualizerProps {
  graphData: any;
  onNodeClick: (node: ConceptNode) => void;
}

export function GraphVisualizer({ graphData, onNodeClick }: GraphVisualizerProps) {
  const { nodes: rawNodes, edges: rawEdges, progress } = graphData;

  const initialNodes = useMemo(() => {
    // Simple topological layout logic
    // We group independent sets (layers)
    const layers = new Map<string, number>();

    const getLayer = (id: string, visited = new Set<string>()): number => {
      if (layers.has(id)) return layers.get(id)!;
      if (visited.has(id)) return 0; // prevent cycle though backend validated DAG
      visited.add(id);

      const incomingEdges = rawEdges.filter((e: any) => e.target === id);
      if (incomingEdges.length === 0) {
        layers.set(id, 0);
        return 0;
      }

      const maxPredLayer = Math.max(...incomingEdges.map((e: any) => getLayer(e.source, visited)));
      const layer = maxPredLayer + 1;
      layers.set(id, layer);
      return layer;
    };

    rawNodes.forEach((n: any) => getLayer(n.id));

    // Group by layer to calculate x positions
    const nodesByLayer: Record<number, any[]> = {};
    rawNodes.forEach((n: any) => {
      const l = layers.get(n.id) || 0;
      if (!nodesByLayer[l]) nodesByLayer[l] = [];
      nodesByLayer[l].push(n);
    });

    const HORIZONTAL_SPACING = 300;
    const VERTICAL_SPACING = 180;

    return rawNodes.map((node: any) => {
      const layer = layers.get(node.id) || 0;
      const indexInLayer = nodesByLayer[layer].indexOf(node);
      const layerSize = nodesByLayer[layer].length;

      const x = (indexInLayer - (layerSize - 1) / 2) * HORIZONTAL_SPACING;
      const y = layer * VERTICAL_SPACING;

      const status = progress?.node_status?.[node.id] || "locked";
      const score = progress?.node_scores?.[node.id];

      return {
        id: node.id,
        type: "knowledge",
        position: { x, y },
        data: { node, status, score },
      };
    });
  }, [rawNodes, rawEdges, progress]);

  const initialEdges = useMemo(() => {
    return rawEdges.map((edge: any) => ({
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      animated: progress?.node_status?.[edge.source] === "mastered" &&
                progress?.node_status?.[edge.target] === "in_progress",
      style: { strokeWidth: 2, opacity: 0.6 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 15,
        height: 15,
      },
    }));
  }, [rawEdges, progress]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  React.useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: any) => {
      onNodeClick(node.data.node);
    },
    [onNodeClick]
  );

  return (
    <div className="w-full h-full min-h-[500px] border rounded-xl overflow-hidden bg-background/50 relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.2}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={24} size={2} color="var(--primary)" opacity={0.1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
