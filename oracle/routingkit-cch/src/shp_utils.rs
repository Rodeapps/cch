use shapefile::dbase::FieldValue;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EdgeAttr {
    pub fid: u64,
    pub u: u64,
    pub v: u64,
    pub length: f64,
    pub highway: Option<String>,
    pub name: Option<String>,
    pub oneway: Option<String>,
    pub maxspeed: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NodeAttr {
    pub osmid: u64,
    pub x: f64,
    pub y: f64,
    pub highway: Option<String>,
    pub r#ref: Option<String>,
}

pub struct GraphArrays {
    pub osmids: Vec<u64>,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub tail: Vec<usize>,
    pub head: Vec<usize>,
    pub weight: Vec<f64>,
}

impl std::fmt::Debug for GraphArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const FRONT: usize = 5;
        const BACK: usize = 5;
        fn slice_fmt<T: std::fmt::Debug>(
            f: &mut std::fmt::Formatter<'_>,
            name: &str,
            data: &[T],
        ) -> std::fmt::Result {
            let len = data.len();
            if len <= FRONT + BACK {
                write!(f, "\n{name}[len={len}] = {:?}", data)
            } else {
                let front = &data[..FRONT];
                let back = &data[len - BACK..];
                write!(
                    f,
                    "\n{name}[len={len}] front={:?} ... back={:?}",
                    front, back
                )
            }
        }
        let (w_min, w_max, w_sum) = self
            .weight
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY, 0f64), |acc, &w| {
                (acc.0.min(w), acc.1.max(w), acc.2 + w)
            });
        let w_avg = if self.weight.is_empty() {
            0.0
        } else {
            w_sum / self.weight.len() as f64
        };
        write!(
            f,
            "GraphArrays summary: nodes={} edges={}",
            self.osmids.len(),
            self.tail.len()
        )?;
        slice_fmt(f, "osmids", &self.osmids)?;
        slice_fmt(f, "x", &self.xs)?;
        slice_fmt(f, "y", &self.ys)?;
        slice_fmt(f, "tail", &self.tail)?;
        slice_fmt(f, "head", &self.head)?;
        slice_fmt(f, "weight", &self.weight)?;
        write!(
            f,
            "\nweight_stats: min={:.3} max={:.3} avg={:.3}",
            w_min, w_max, w_avg
        )?;
        Ok(())
    }
}

trait RecordExt {
    fn fv(&self, key: &str) -> Option<&FieldValue>;
    fn num(&self, key: &str) -> Option<u64>;
    fn f64v(&self, key: &str) -> Option<f64>;
    fn strv(&self, key: &str) -> Option<String>;
    fn must_num(&self, key: &str, idx: usize, kind: &str) -> Result<u64, String>;
    fn must_f64(&self, key: &str, idx: usize, kind: &str) -> Result<f64, String>;
}
impl RecordExt for shapefile::dbase::Record {
    fn fv(&self, key: &str) -> Option<&FieldValue> {
        self.get(key)
    }
    fn num(&self, key: &str) -> Option<u64> {
        self.fv(key).and_then(|v| match v {
            FieldValue::Numeric(opt) => opt.map(|f| f as u64),
            FieldValue::Character(Some(s)) => s.parse().ok(),
            _ => None,
        })
    }
    fn f64v(&self, key: &str) -> Option<f64> {
        self.fv(key).and_then(|v| match v {
            FieldValue::Numeric(opt) => opt.map(|f| f as f64),
            FieldValue::Character(Some(s)) => s.parse().ok(),
            _ => None,
        })
    }
    fn strv(&self, key: &str) -> Option<String> {
        self.fv(key).and_then(|v| match v {
            FieldValue::Character(Some(s)) => Some(s.trim().to_string()),
            _ => None,
        })
    }
    fn must_num(&self, key: &str, idx: usize, kind: &str) -> Result<u64, String> {
        self.num(key)
            .ok_or_else(|| format!("Missing required field '{key}' at {kind} record {idx}"))
    }
    fn must_f64(&self, key: &str, idx: usize, kind: &str) -> Result<f64, String> {
        self.f64v(key)
            .ok_or_else(|| format!("Missing required field '{key}' at {kind} record {idx}"))
    }
}

pub fn load_edges<P: AsRef<Path>>(path: &P) -> Result<Vec<EdgeAttr>, Box<dyn std::error::Error>> {
    let mut reader = shapefile::Reader::from_path(path)?;
    let mut edges = Vec::new();
    let mut idx = 0usize;
    for rec in reader.iter_shapes_and_records() {
        let (_shape, record) = rec?;
        let fid = record.must_num("fid", idx, "edge")?;
        let u = record.must_num("u", idx, "edge")?;
        let v = record.must_num("v", idx, "edge")?;
        let length = record.must_f64("length", idx, "edge")?;
        edges.push(EdgeAttr {
            fid,
            u,
            v,
            length,
            highway: record.strv("highway"),
            name: record.strv("name"),
            oneway: record.strv("oneway"),
            maxspeed: record.strv("maxspeed"),
        });
        idx += 1;
    }
    Ok(edges)
}

pub fn load_nodes<P: AsRef<Path>>(path: &P) -> Result<Vec<NodeAttr>, Box<dyn std::error::Error>> {
    let mut reader = shapefile::Reader::from_path(path)?;
    let mut nodes = Vec::new();
    let mut idx = 0usize;
    for rec in reader.iter_shapes_and_records() {
        let (shape, record) = rec?;
        let (x, y) = match shape {
            shapefile::Shape::Point(p) => (p.x, p.y),
            shapefile::Shape::PointZ(p) => (p.x, p.y),
            _ => continue,
        };
        let osmid = record.must_num("osmid", idx, "node")?;
        let highway = record.strv("highway");
        let r#ref = record.strv("ref");
        nodes.push(NodeAttr {
            osmid,
            x,
            y,
            highway,
            r#ref,
        });
        idx += 1;
    }
    Ok(nodes)
}

pub fn build_graph_arrays(nodes: &[NodeAttr], edges: &[EdgeAttr]) -> Result<GraphArrays, String> {
    use std::collections::HashMap;
    let mut id_map = HashMap::with_capacity(nodes.len());
    let mut osmids = Vec::with_capacity(nodes.len());
    let mut xs = Vec::with_capacity(nodes.len());
    let mut ys = Vec::with_capacity(nodes.len());
    for (i, n) in nodes.iter().enumerate() {
        if id_map.insert(n.osmid, i).is_some() {
            return Err(format!("Duplicate osmid {}", n.osmid));
        }
        osmids.push(n.osmid);
        xs.push(n.x);
        ys.push(n.y);
    }
    let mut tail = Vec::with_capacity(edges.len());
    let mut head = Vec::with_capacity(edges.len());
    let mut weight = Vec::with_capacity(edges.len());
    for e in edges {
        let &tu = id_map
            .get(&e.u)
            .ok_or_else(|| format!("Edge u osmid {} not found", e.u))?;
        let &hv = id_map
            .get(&e.v)
            .ok_or_else(|| format!("Edge v osmid {} not found", e.v))?;
        tail.push(tu);
        head.push(hv);
        weight.push(e.length);
    }
    Ok(GraphArrays {
        osmids,
        xs,
        ys,
        tail,
        head,
        weight,
    })
}
