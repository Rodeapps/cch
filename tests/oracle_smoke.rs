#[test]
fn oracle_links_and_builds_tiny_cch() {
    use routingkit_cch::ffi;
    let order: Vec<u32> = (0..4).collect();
    let tail = vec![0u32,1,2]; let head = vec![1u32,2,3];
    let cch = unsafe { ffi::cch_new(&order, &tail, &head, |_| {}, false) };
    assert_eq!(unsafe { ffi::cch_node_count(cch.as_ref().unwrap()) }, 4);
}
