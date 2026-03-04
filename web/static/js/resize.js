/**
 * Panel resize handles — drag to resize columns and rows.
 *
 * Horizontal handles resize grid-template-columns on .workspace.
 * Vertical handles resize flex-basis on sub-panels within columns.
 */

export function init() {
  const workspace = document.getElementById('workspace');

  // Horizontal column resizers
  setupColumnResize('resize-left-h', workspace, 'left');
  setupColumnResize('resize-right-h', workspace, 'right');

  // Vertical row resizers
  setupRowResize('resize-left-v', 'left-top', 'left-bottom');
  setupRowResize('resize-right-v', 'right-top', 'right-bottom');
}

function setupColumnResize(handleId, workspace, side) {
  const handle = document.getElementById(handleId);
  if (!handle) return;

  let startX, startLeftW, startRightW, totalW;

  const colLeft = document.getElementById('col-left');
  const colRight = document.getElementById('col-right');

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startX = e.clientX;

    const leftRect = colLeft.getBoundingClientRect();
    const rightRect = colRight.getBoundingClientRect();
    const workRect = workspace.getBoundingClientRect();

    startLeftW = leftRect.width;
    startRightW = rightRect.width;
    totalW = workRect.width;

    handle.classList.add('active');
    document.body.classList.add('resizing');

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  function onMove(e) {
    const dx = e.clientX - startX;
    const minCol = 180;
    const minCenter = 300;
    // 12px total for two 6px handles
    const handleSpace = 12;

    if (side === 'left') {
      let newLeft = Math.max(minCol, startLeftW + dx);
      // Ensure center has enough room
      const maxLeft = totalW - handleSpace - startRightW - minCenter;
      newLeft = Math.min(newLeft, maxLeft);
      workspace.style.gridTemplateColumns = `${newLeft}px 6px 1fr 6px ${startRightW}px`;
    } else {
      let newRight = Math.max(minCol, startRightW - dx);
      const maxRight = totalW - handleSpace - startLeftW - minCenter;
      newRight = Math.min(newRight, maxRight);
      workspace.style.gridTemplateColumns = `${startLeftW}px 6px 1fr 6px ${newRight}px`;
    }

    // Notify viewer to resize canvas
    window.dispatchEvent(new Event('resize'));
  }

  function onUp() {
    handle.classList.remove('active');
    document.body.classList.remove('resizing');
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  }
}

function setupRowResize(handleId, topId, bottomId) {
  const handle = document.getElementById(handleId);
  const topEl = document.getElementById(topId);
  const bottomEl = document.getElementById(bottomId);
  if (!handle || !topEl || !bottomEl) return;

  let startY, startTopH, startBottomH;

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startY = e.clientY;
    startTopH = topEl.getBoundingClientRect().height;
    startBottomH = bottomEl.getBoundingClientRect().height;

    handle.classList.add('active');
    document.body.classList.add('resizing-v');

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });

  function onMove(e) {
    const dy = e.clientY - startY;
    const total = startTopH + startBottomH;
    const min = 80;

    let newTopH = Math.max(min, Math.min(total - min, startTopH + dy));
    let newBottomH = total - newTopH;

    topEl.style.flex = `0 0 ${newTopH}px`;
    bottomEl.style.flex = `0 0 ${newBottomH}px`;

    // Notify viewer to resize canvas
    window.dispatchEvent(new Event('resize'));
  }

  function onUp() {
    handle.classList.remove('active');
    document.body.classList.remove('resizing-v');
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  }
}
