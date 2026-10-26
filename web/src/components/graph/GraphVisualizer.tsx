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
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { ConceptNode } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS_STYLES: Record<string, { bg: string; border: string; text: string; dot: string; glow: string }> = {
  locked:      { bg: "bg-[hsl(215,15%,13%)]",   border: "border-[hsl(215,15%,22%)]",  text: "text-[hsl(215,15%,42%)]",  dot: "bg-[hsl(215,15%,38%)]",   glow: "" },
  available:   { bg: "bg-[hsl(38,60%,12%)]",    border: "border-[hsl(38,95%,45%,0.5)]", text: "text-[hsl(38,90%,72%)]",  dot: "bg-[hsl(38,95%,55%)]",    glow: "shadow-[0_0_16px_hsl(38,95%,55%,0.2)]" },
  in_progress: { bg: "bg-[hsl(174,50%,10%)]",   border: "border-[hsl(174,70%,42%,0.6)]", text: "text-[hsl(174,70%,62%)]", dot: "bg-[hsl(174,70%,42%)]",  glow: "shadow-[0_0_20px_hsl(174,70%,42%,0.3)]" },
  mastered:    { bg: "bg-[hsl(142,60%,9%)]",    border: "border-[hsl(142,72%,38%,0.5)]", text: "text-[hsl(142,72%,60%)]", dot: "bg-[hsl(142,72%,45%)]", glow: "shadow-[0_0_16px_hsl(142,72%,45%,0.2)]" },
};

const STATUS_LABEL: Record<string, string> = {
  locked: "Locked", available: "Available", in_progress: "In Progress", mastered: "Mastered",
};

interface KnowledgeNodeData {
  node: ConceptNode;
  status: string;
  score?: number;
  isActive?: boolean;
}

const KnowledgeNode = ({ data, selected }: NodeProps) => {
  const { node, status, score, isActive } = data as unknown as KnowledgeNodeData;
  const s = STATUS_STYLES[status] || STATUS_STYLES.locked;

  return (
    <div
      className={cn(
        "relative min-w-[210px] max-w-[250px] rounded-xl border-2 p-3.5 transition-all duration-200 cursor-pointer",
        s.bg, s.border, s.glow,
        isActive && "ring-2 ring-[hsl(38,95%,55%)] ring-offset-2 ring-offset-[hsl(220,25%,6%)] shadow-[0_0_24px_hsl(38,95%,55%,0.35)] scale-[1.03]",
        selected && !isActive && "scale-105 ring-2 ring-white/20 ring-offset-1 ring-offset-transparent"
      )}
    >
      <Handle type="target" position={Position.Top} style={{ background: "hsl(220,18%,30%)", border: "2px solid hsl(220,18%,40%)", width: 10, height: 10 }} />

      <div className="flex items-start gap-2 mb-2">
        <span className={cn("mt-1 w-2.5 h-2.5 rounded-full shrink-0", s.dot,
          (status === "in_progress" || isActive) && "animate-pulse")} />
        <div className="min-w-0 flex-1">
          <h3 className={cn("font-semibold text-sm leading-tight truncate", s.text)}>{node.title}</h3>
          {isActive && (
            <span className="text-[10px] font-bold text-[hsl(38,90%,70%)] tracking-wider uppercase">
              ✦ Current Focus
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 pt-1 border-t border-[hsl(220,18%,18%)]">
        <span className={cn("tag text-[9px]", status === "mastered" ? "status-mastered" : status === "in_progress" ? "status-in_progress" : status === "available" ? "status-available" : "status-locked")}>
          {STATUS_LABEL[status] || "Locked"}
        </span>
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] text-muted-foreground font-mono">Lvl {node.difficulty}</span>
          {score !== undefined && (
            <span className={cn("text-[9px] font-bold font-mono", s.text)}>{Math.round(score * 100)}%</span>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} style={{ background: "hsl(220,18%,30%)", border: "2px solid hsl(220,18%,40%)", width: 10, height: 10 }} />
    </div>
  );
};

const nodeTypes = { knowledge: KnowledgeNode };

interface GraphVisualizerProps {
  graphData: any;
  onNodeClick: (node: ConceptNode) => void;
  activeNodeId?: string | null;
  className?: string;
}

export function GraphVisualizer({ graphData, onNodeClick, activeNodeId, className }: GraphVisualizerProps) {
  const { nodes: rawNodes, edges: rawEdges, progress } = graphData;

  const initialNodes = useMemo(() => {
    const layers = new Map<string, number>();

    const getLayer = (id: string, visited = new Set<string>()): number => {
      if (layers.has(id)) return layers.get(id)!;
      if (visited.has(id)) return 0;
      visited.add(id);
      const inc = rawEdges.filter((e: any) => e.target === id);
      if (inc.length === 0) { layers.set(id, 0); return 0; }
      const layer = Math.max(...inc.map((e: any) => getLayer(e.source, visited))) + 1;
      layers.set(id, layer);
      return layer;
    };

    rawNodes.forEach((n: any) => getLayer(n.id));

    const nodesByLayer: Record<number, any[]> = {};
    rawNodes.forEach((n: any) => {
      const l = layers.get(n.id) || 0;
      if (!nodesByLayer[l]) nodesByLayer[l] = [];
      nodesByLayer[l].push(n);
    });

    const H = 270, V = 160;

    return rawNodes.map((node: any) => {
      const layer = layers.get(node.id) || 0;
      const idx = nodesByLayer[layer].indexOf(node);
      const size = nodesByLayer[layer].length;
      return {
        id: node.id,
        type: "knowledge",
        position: { x: (idx - (size - 1) / 2) * H, y: layer * V },
        data: {
          node,
          status: progress?.node_status?.[node.id] || "locked",
          score: progress?.node_scores?.[node.id],
          isActive: node.id === activeNodeId,
        },
      };
    });
  }, [rawNodes, rawEdges, progress, activeNodeId]);

  const initialEdges = useMemo(() => rawEdges.map((edge: any) => {
    const srcMastered = progress?.node_status?.[edge.source] === "mastered";
    const tgtActive = progress?.node_status?.[edge.target] === "in_progress" || edge.target === activeNodeId;
    return {
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      animated: srcMastered && tgtActive,
      style: {
        strokeWidth: (srcMastered || tgtActive) ? 2.5 : 1.5,
        stroke: (srcMastered || tgtActive) ? "hsl(38,95%,55%,0.7)" : "hsl(220,18%,28%)",
        opacity: srcMastered ? 0.95 : 0.5,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 12, height: 12,
        color: (srcMastered || tgtActive) ? "hsl(38,95%,55%)" : "hsl(220,18%,38%)",
      },
    };
  }), [rawEdges, progress, activeNodeId]);

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  React.useEffect(() => { setNodes(initialNodes); setEdges(initialEdges); }, [initialNodes, initialEdges]);

  const handleNodeClick = useCallback((_: React.MouseEvent, node: any) => {
    onNodeClick(node.data.node);
  }, [onNodeClick]);

  return (
    <div className={cn("relative w-full h-full select-none", className)}
      style={{ background: "hsl(220, 25%, 6%)" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2, minZoom: 0.35, maxZoom: 1.2 }}
        minZoom={0.15}
        maxZoom={2.5}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={28} size={1.2} color="hsl(220,18%,22%)" />
        <Controls showInteractive={false} className="m-3" />
        <MiniMap
          nodeColor={(n) => {
            const st = (n.data as any)?.status;
            return st === "mastered" ? "hsl(142,72%,40%)" : st === "in_progress" ? "hsl(174,70%,42%)" : st === "available" ? "hsl(38,95%,50%)" : "hsl(215,15%,25%)";
          }}
          maskColor="hsl(220,25%,6%,0.85)"
          style={{ background: "hsl(222,23%,10%)", borderRadius: 8, border: "1px solid hsl(220,18%,18%)" }}
          className="m-3"
        />
      </ReactFlow>
    </div>
  );
}
